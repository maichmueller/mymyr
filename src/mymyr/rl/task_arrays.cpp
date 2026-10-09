#include "mymyr/rl/task_arrays.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/rl/expand.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mymyr::rl
{
const ArrayInfo* ArrayBundle::find(std::string_view name) const noexcept
{
    for (const ArrayInfo& a : m_arrays)
        if (a.name == name)
            return &a;
    return nullptr;
}

i64 ArrayBundle::scalar(std::string_view name, i64 fallback) const noexcept
{
    for (const auto& [n, v] : m_scalars)
        if (n == name)
            return v;
    return fallback;
}

void ArrayBundle::Builder::add_bytes(std::string name, DType dtype, std::vector<i64> shape, std::vector<std::byte> bytes,
                                     bool words)
{
    ArrayInfo info;
    info.name = std::move(name);
    info.dtype = dtype;
    info.shape = std::move(shape);
    info.words = words;
    if (info.elements() * dtype_bytes(dtype) != bytes.size())
        throw std::logic_error("ArrayBundle: array '" + info.name + "' has a size that does not match its shape");
    m_items.emplace_back(std::move(info), std::move(bytes));
}

ArrayBundle ArrayBundle::Builder::build()
{
    ArrayBundle b;
    u64 off = 0;
    for (auto& [info, bytes] : m_items)
    {
        info.offset = off;
        off += (bytes.size() + 63) & ~u64{63};
    }
    b.m_bytes = std::max<u64>(off, 64);
    // 64-byte aligned block (zero-copy importers such as XLA's CPU client want aligned buffers): over-allocate and
    // align by hand
    b.m_block = std::make_unique<std::byte[]>(b.m_bytes + 64);
    const auto addr = reinterpret_cast<std::uintptr_t>(b.m_block.get());
    b.m_base = b.m_block.get() + ((64 - addr % 64) % 64);
    for (auto& [info, bytes] : m_items)
    {
        if (!bytes.empty())
            std::memcpy(b.m_base + info.offset, bytes.data(), bytes.size());
        b.m_arrays.push_back(std::move(info));
    }
    b.m_scalars = std::move(m_scalars);
    m_items.clear();
    return b;
}

namespace
{
/// Slots [0, count) of one kind with their canonical ids: every slot in frozen mode; under lazy slots the published
/// prefix (a slot is published by the release store of slot_of[cid] after its record was written).
struct Slots
{
    u32 count = 0;
    std::vector<u64> cid;
};

Slots published(const Task& task, AtomKind kind)
{
    const AtomIndex& A = task.atoms();
    const CanonicalLayout& L = A.layout();
    const u64 first = kind == AtomKind::Fluent ? 0 : L.fluent_count;
    const u64 last = kind == AtomKind::Fluent ? L.fluent_count : L.total;
    Slots s;
    if (A.mode() == AtomMode::Frozen)
    {
        s.count = static_cast<u32>(last - first);
        s.cid.resize(s.count);
        for (u32 i = 0; i < s.count; ++i)
            s.cid[i] = first + i;
        return s;
    }
    const u32 n = kind == AtomKind::Fluent ? A.fluent_slots() : A.derived_slots();
    constexpr u64 k_missing = ~u64{0};
    s.cid.assign(n, k_missing);
    const std::atomic<u32>* table = A.slot_table();
    for (u64 c = first; c < last; ++c)
    {
        const u32 v = table[c].load(std::memory_order_acquire);
        if (v < AtomIndex::k_pending && v < n)
            s.cid[v] = c;
    }
    s.count = n;
    for (u32 i = 0; i < n; ++i)
        if (s.cid[i] == k_missing)
        {
            s.count = i;  // an intern in flight on another thread: stop before it
            break;
        }
    s.cid.resize(s.count);
    return s;
}

std::vector<i64> shape1(u64 n) { return {static_cast<i64>(n)}; }
std::vector<i64> shape2(u64 n, u64 m) { return {static_cast<i64>(n), static_cast<i64>(m)}; }

i32 pred_kind_code(formalism::PredKind k)
{
    return k == formalism::PredKind::Static ? k_pred_static : k == formalism::PredKind::Fluent ? k_pred_fluent : k_pred_derived;
}

/// Atom records of a slot range as pred [n], CSR args, padded args [n, A].
void add_atoms(ArrayBundle::Builder& b, const Task& task, AtomKind kind, const Slots& s, const std::string& prefix,
               bool csr)
{
    const AtomIndex& A = task.atoms();
    const auto& arity = task.compiled().arity;
    u32 max_arity = 0;
    for (u32 p = 0; p < arity.size(); ++p)
        if (task.compiled().kinds[p] == (kind == AtomKind::Fluent ? formalism::PredKind::Fluent : formalism::PredKind::Derived))
            max_arity = std::max(max_arity, arity[p]);
    std::vector<i32> pred(s.count), offsets(s.count + 1, 0), args, padded(static_cast<usize>(s.count) * max_arity, -1);
    for (u32 i = 0; i < s.count; ++i)
    {
        const u32* rec = A.record(kind, i);
        pred[i] = static_cast<i32>(rec[0]);
        const u32 k = arity[rec[0]];
        for (u32 j = 0; j < k; ++j)
        {
            args.push_back(static_cast<i32>(rec[1 + j]));
            padded[static_cast<usize>(i) * max_arity + j] = static_cast<i32>(rec[1 + j]);
        }
        offsets[i + 1] = static_cast<i32>(args.size());
    }
    b.add(prefix + "_pred", DType::I32, shape1(s.count), pred);
    if (csr)
    {
        b.add(prefix + "_args", DType::I32, shape1(args.size()), args);
        b.add(prefix + "_args_offsets", DType::I32, shape1(s.count + 1), offsets);
    }
    b.add(prefix + "_args_padded", DType::I32, shape2(s.count, max_arity), padded);
    std::vector<i32> cid(s.count);
    for (u32 i = 0; i < s.count; ++i)
        cid[i] = static_cast<i32>(s.cid[i]);  // canonical spaces are below 2^28 (CanonicalLayout::k_max_total)
    b.add(prefix + "_cid", DType::I32, shape1(s.count), cid);
}

void add_metadata(ArrayBundle::Builder& b, const Task& task, const Slots& fluent, const Slots& derived)
{
    const plan::Compiled& C = task.compiled();
    const formalism::TaskData& D = task.data();
    const u32 P = static_cast<u32>(C.arity.size());

    add_atoms(b, task, AtomKind::Fluent, fluent, "atom", true);
    add_atoms(b, task, AtomKind::Derived, derived, "derived", false);

    // fluent slots grouped by predicate (ascending slot within a group)
    std::vector<i32> group_offsets(P + 1, 0), group(fluent.count);
    for (u32 i = 0; i < fluent.count; ++i)
        ++group_offsets[task.atoms().record(AtomKind::Fluent, i)[0] + 1];
    for (u32 p = 0; p < P; ++p)
        group_offsets[p + 1] += group_offsets[p];
    {
        std::vector<i32> fill(group_offsets.begin(), group_offsets.end() - 1);
        for (u32 i = 0; i < fluent.count; ++i)
            group[fill[task.atoms().record(AtomKind::Fluent, i)[0]]++] = static_cast<i32>(i);
    }
    b.add("pred_slots", DType::I32, shape1(fluent.count), group);
    b.add("pred_slot_offsets", DType::I32, shape1(P + 1), group_offsets);

    std::vector<i32> parity(P), pkind(P);
    for (u32 p = 0; p < P; ++p)
    {
        parity[p] = static_cast<i32>(C.arity[p]);
        pkind[p] = pred_kind_code(C.kinds[p]);
    }
    b.add("pred_arity", DType::I32, shape1(P), parity);
    b.add("pred_kind", DType::I32, shape1(P), pkind);

    // static atoms (types included: normalization compiles them into static unary predicates)
    u32 smax = 0;
    for (const auto& a : D.static_init)
        smax = std::max(smax, a.objects.count);
    std::vector<i32> spred(D.static_init.size()), soff(D.static_init.size() + 1, 0), sargs,
        spad(D.static_init.size() * smax, -1);
    for (usize i = 0; i < D.static_init.size(); ++i)
    {
        const auto& a = D.static_init[i];
        spred[i] = static_cast<i32>(a.pred.v);
        u32 j = 0;
        for (ObjectId o : D.objects_of(a))
        {
            sargs.push_back(static_cast<i32>(o.v));
            spad[i * smax + j++] = static_cast<i32>(o.v);
        }
        soff[i + 1] = static_cast<i32>(sargs.size());
    }
    b.add("static_pred", DType::I32, shape1(spred.size()), spred);
    b.add("static_args", DType::I32, shape1(sargs.size()), sargs);
    b.add("static_args_offsets", DType::I32, shape1(soff.size()), soff);
    b.add("static_args_padded", DType::I32, shape2(spred.size(), smax), spad);

    u32 max_arity = 0;
    for (u32 a : C.arity)
        max_arity = std::max(max_arity, a);
    b.scalar("num_objects", C.num_objects);
    b.scalar("num_predicates", P);
    b.scalar("num_atoms", fluent.count);
    b.scalar("num_derived_atoms", derived.count);
    b.scalar("num_static_atoms", static_cast<i64>(D.static_init.size()));
    b.scalar("max_atoms", static_cast<i64>(C.layout.fluent_count));
    b.scalar("max_derived_atoms", static_cast<i64>(C.layout.total - C.layout.fluent_count));
    b.scalar("words", bits::words_for(fluent.count));
    b.scalar("max_words", task.max_words());
    b.scalar("max_arity", max_arity);
    b.scalar("atom_mode", task.atoms().mode() == AtomMode::Frozen ? 1 : 0);
}

std::vector<u64> padded_words(std::span<const u64> w, u32 W)
{
    std::vector<u64> out(W, 0);
    std::copy_n(w.begin(), std::min<usize>(w.size(), W), out.begin());
    return out;
}

/// Section "plan" of version 2 (task_arrays_view.hpp): the compiled plans with every pointer turned into an offset.
/// Tables are appended in a fixed traversal order (schemas: pre[0], pre[1], conditional effects; then strata; then
/// the goal), so the export is deterministic.
class PlanWriter
{
public:
    PlanWriter(const Task& task, u32 fluent_slots, u32 derived_slots)
        : m_task(task), C(task.compiled()), m_fluent(fluent_slots), m_derived(derived_slots), m_costs(device_search_costs(task))
    {
    }

    void write(ArrayBundle::Builder& b)
    {
        const u32 P = static_cast<u32>(C.arity.size());
        compact_view();
        // position tables: the canonical layout's, then every static relation's
        m_rs = C.layout.rs;
        std::vector<u64> static_rel(static_cast<usize>(P) * dev::k_sr_count, 0);
        std::vector<u64> sbits, stable;
        m_static_rs.assign(P, 0);
        for (u32 p = 0; p < P; ++p)
        {
            u64* r = static_rel.data() + static_cast<usize>(p) * dev::k_sr_count;
            r[dev::k_sr_bits] = r[dev::k_sr_table] = dev::k_none64;
            if (C.kinds[p] != formalism::PredKind::Static)
                continue;
            const plan::StaticRelation& R = C.statics[p];
            m_static_rs[p] = m_rs.size();
            m_rs.insert(m_rs.end(), R.rs.begin(), R.rs.end());
            r[dev::k_sr_arity] = R.arity;
            r[dev::k_sr_size] = R.size;
            r[dev::k_sr_rs] = m_static_rs[p];
            r[dev::k_sr_mask] = R.mask;
            if (!R.bits.empty())
            {
                r[dev::k_sr_bits] = sbits.size();
                sbits.insert(sbits.end(), R.bits.begin(), R.bits.end());
            }
            else
            {
                r[dev::k_sr_table] = stable.size();
                stable.insert(stable.end(), R.table.begin(), R.table.end());
            }
        }
        // pattern variables keep the indices of plan::Compiled::pattern_vars; their tables become offsets into plan_rs
        m_vars.assign(C.pattern_vars.size() * 2, 0);

        std::vector<u32> schema(C.schemas.size() * dev::k_sc_count, 0);
        for (usize s = 0; s < C.schemas.size(); ++s)
        {
            const plan::Schema& ps = C.schemas[s];
            u32* row = schema.data() + s * dev::k_sc_count;
            row[dev::k_sc_arity] = ps.arity;
            row[dev::k_sc_bind_size] = ps.bind_size;
            // forward checking sorts the bindings: a matcher whose fixed order binds in parameter order keeps it where
            // the packed keys of the env's picks would not fit 64 bits (cuda::lifted::pick_keys_fit)
            u32 bits = 1;
            while (bits < 32 && (u64{1} << bits) < C.num_objects)
                ++bits;
            const bool keys_fit = u64{ps.arity} * bits <= 64;
            for (u32 wi = 0; wi < 2; ++wi)
            {
                // deep matchers (the device's deep kernels) keep the plan's matcher: their searches vary from state to
                // state far more than the probe's few states show (organic-synthesis)
                const plan::Matcher& m = ps.pre[wi];
                const bool deep = m.total > dev::k_max_matcher_depth || m.steps.size() > dev::k_max_matcher_depth;
                const bool fc = deep ? m.use_fc : prefers_fc(m_costs.schemas[s][wi]) && (keys_fit || !fixed_in_order(m));
                row[wi == 0 ? dev::k_sc_pre0 : dev::k_sc_pre1] = matcher(m, fc);
            }
            put(row, dev::k_sc_adds, patterns(ps.adds));
            put(row, dev::k_sc_dels, patterns(ps.dels));
            put(row, dev::k_sc_pre_lits, checks(ps.pre_lits));
            row[dev::k_sc_ces] = count(m_cond_effect.size() / 5);
            row[dev::k_sc_ces_n] = count(ps.ces.size());
            // the condition matchers first, so that a schema's conditional effects are contiguous rows
            std::vector<std::array<u32, 5>> ces;
            for (const plan::CondEffect& ce : ps.ces)
            {
                const u32 m = matcher(ce.cond, false);
                const auto [a, an] = patterns(ce.adds);
                const auto [d, dn] = patterns(ce.dels);
                ces.push_back({m, a, an, d, dn});
            }
            for (const auto& ce : ces)
                m_cond_effect.insert(m_cond_effect.end(), ce.begin(), ce.end());
        }
        std::vector<u32> axiom, stratum;
        for (const plan::Stratum& st : C.strata)
        {
            stratum.insert(stratum.end(), {count(axiom.size() / 2), count(st.axioms.size()), st.recursive ? 1u : 0u});
            for (const plan::Axiom& x : st.axioms)
            {
                const u32 head = pattern(x.head);
                const u32 body = matcher(x.body, prefers_fc(m_costs.axioms[axiom.size() / 2]));
                axiom.insert(axiom.end(), {head, body});
            }
        }
        const auto [goal_check, goal_check_n] = checks(C.goal.lits);

        // atom index snapshot: the slot table and the view ops of the slots covered by the export
        const AtomIndex& A = m_task.atoms();
        const CanonicalLayout& L = C.layout;
        std::vector<u32> slot_of(L.total + 1, dev::k_none);
        const std::atomic<u32>* table = A.slot_table();
        for (u64 c = 0; c < L.total; ++c)
        {
            const u32 v = table[c].load(std::memory_order_acquire);
            const u32 limit = c < L.fluent_count ? m_fluent : m_derived;
            if (v < AtomIndex::k_pending && v < limit)
                slot_of[c] = v;
        }
        // the view ops in the device's compact view (rows no matcher reads set nothing)
        auto ops = [&](AtomKind k, u32 n)
        {
            std::vector<u32> out(static_cast<usize>(n) * 4);
            for (u32 s = 0; s < n; ++s)
            {
                const ViewOp& op = A.view_op(k, s);
                const u32 r1 = op.row1 == ViewOp::k_none || !m_view_live[op.row1] ? dev::k_none : m_view_map[op.row1];
                const u32 r2 = op.row2 == ViewOp::k_none || !m_view_live[op.row2] ? dev::k_none : m_view_map[op.row2];
                out[s * usize{4} + 0] = r1;
                out[s * usize{4} + 1] = r1 == dev::k_none ? 0 : op.bit1;
                out[s * usize{4} + 2] = r2;
                out[s * usize{4} + 3] = r2 == dev::k_none ? 0 : op.bit2;
            }
            return out;
        };

        b.add("plan_slot_of", DType::U32, shape1(slot_of.size()), slot_of);
        b.add("plan_rs", DType::U64, shape1(m_rs.size()), m_rs);
        b.add("plan_static_words", DType::U64, shape1(C.static_words.size()), C.static_words);
        b.add("plan_static_rel", DType::U64, shape2(P, dev::k_sr_count), static_rel);
        b.add("plan_static_bits", DType::U64, shape1(sbits.size()), sbits);
        b.add("plan_static_table", DType::U64, shape1(stable.size()), stable);
        b.add("plan_fluent_view_op", DType::U32, shape2(m_fluent, 4), ops(AtomKind::Fluent, m_fluent));
        b.add("plan_derived_view_op", DType::U32, shape2(m_derived, 4), ops(AtomKind::Derived, m_derived));
        b.add("plan_view_map", DType::U32, shape1(m_view_map.size()), m_view_map);
        b.add("plan_pattern", DType::U32, shape2(m_pattern_base.size(), 4), m_pattern_rows);
        b.add("plan_pattern_base", DType::U64, shape1(m_pattern_base.size()), m_pattern_base);
        b.add("plan_pattern_var", DType::U32, shape2(m_vars.size() / 2, 2), m_vars);
        b.add("plan_check", DType::U32, shape2(m_check_rows.size() / 2, 2), m_check_rows);
        b.add("plan_matcher", DType::U32, shape2(m_matcher_rows.size() / dev::k_mc_count, dev::k_mc_count), m_matcher_rows);
        b.add("plan_dom0", DType::U64, shape1(m_dom0.size()), m_dom0);
        b.add("plan_index", DType::U32, shape1(m_index.size()), m_index);
        b.add("plan_unary", DType::U64, shape2(m_unary.size() / 2, 2), m_unary);
        b.add("plan_row", DType::U64, shape2(m_rows.size() / 3, 3), m_rows);
        b.add("plan_edge", DType::U64, shape2(m_edges.size() / 3, 3), m_edges);
        b.add("plan_step", DType::U32, shape2(m_steps.size() / 5, 5), m_steps);
        b.add("plan_schema", DType::U32, shape2(C.schemas.size(), dev::k_sc_count), schema);
        b.add("plan_cond_effect", DType::U32, shape2(m_cond_effect.size() / 5, 5), m_cond_effect);
        b.add("plan_axiom", DType::U32, shape2(axiom.size() / 2, 2), axiom);
        b.add("plan_stratum", DType::U32, shape2(stratum.size() / 3, 3), stratum);

        if (C.num.slots)
            write_numeric(b, schema);

        b.scalar("section_plan", k_section_plan_version);
        b.scalar("plan_max_bind", C.max_bind);
        b.scalar("plan_view_rows", m_view_rows);
        b.scalar("plan_atom_total", static_cast<i64>(L.total));
        b.scalar("plan_fluent_total", static_cast<i64>(L.fluent_count));
        b.scalar("plan_fluent_slots", m_fluent);
        b.scalar("plan_derived_slots", m_derived);
        b.scalar("plan_goal_check", goal_check);
        b.scalar("plan_goal_check_n", goal_check_n);
        b.scalar("plan_has_conditional_effects", C.has_conditional_effects ? 1 : 0);
        b.scalar("plan_has_axioms", C.strata.empty() ? 0 : 1);
    }

private:
    void write_numeric(ArrayBundle::Builder& b, const std::vector<u32>& schema);

    static u32 count(u64 v)
    {
        if (v >= dev::k_none)
            throw std::length_error("mymyr: device_arrays: a plan table exceeds 2^32 - 2 entries");
        return static_cast<u32>(v);
    }

    /// The device's compact view: of the rows of the CPU's ViewLayout, only those that an exported matcher reads
    /// (a schema's two matchers, the conditions of its conditional effects, the axiom bodies) and that some atom can
    /// set. A unary reference reads its one row; a reference indexed by a bound object (plan::Row, plan::Edge) reads
    /// row base + o, which some atom can set only if o lies in the canonical layout's domain of the indexing position
    /// (forward rows: position 0, backward rows: position 1). The kept rows are renumbered in order; the rows a matcher
    /// reads that no atom sets share one zero row (the last). m_view_map maps every ViewLayout row to its device row
    /// (k_none: no matcher reads it) and is exported as plan_view_map: the matchers' TableRefs keep the CPU's rows and
    /// the device reads row view_map[ref + o]. Views are built, stored and read per state by every device kernel,
    /// while an object-indexed table is mostly rows of objects of other types: philosophers-6 keeps 84 of its 509
    /// rows, schedule 12 of 210.
    void compact_view()
    {
        const ViewLayout& V = C.view;
        const CanonicalLayout& L = C.layout;
        const u32 R = V.rows;
        const u32 n = C.num_objects;
        // live[r]: some atom can set row r (its predicate has atoms, and the object indexing it lies in the domain)
        std::vector<u8> live(R, 0);
        const u32 P = static_cast<u32>(C.arity.size());
        for (u32 p = 0; p < P; ++p)
        {
            if (p >= L.offset.size() || L.offset[p] == CanonicalLayout::k_none || L.size[p] == 0)
                continue;
            if (V.unary_row[p] != ViewLayout::k_none && V.unary_row[p] < R)
                live[V.unary_row[p]] = 1;
            for (u32 pos = 0; pos < 2; ++pos)
            {
                const u32 first = pos == 0 ? V.fwd_row[p] : V.bwd_row[p];
                if (first == ViewLayout::k_none || L.arity[p] != 2)
                    continue;
                const u64* rs = L.position_table(p, pos);
                for (u32 o = 0; o < n && first + o < R; ++o)
                    if (rs[o] != CanonicalLayout::k_outside)
                        live[first + o] = 1;
            }
        }
        std::vector<u8> read(R, 0);
        auto one = [&](plan::TableRef r)
        {
            if (r.is_view() && r.value() < R)
                read[r.value()] = 1;
        };
        auto block = [&](plan::TableRef r)
        {
            if (r.is_view())
                for (u64 i = r.value(); i < r.value() + n && i < R; ++i)
                    read[i] = 1;
        };
        auto mark = [&](const plan::Matcher& m)
        {
            for (const plan::Unary& u : m.unary)
                one(u.ptr);
            for (const plan::Row& r : m.rows)
                block(r.base);
            for (const plan::Edge& e : m.fc_out)
                block(e.base);
            for (const plan::Row& r : m.fc_pre)
                block(r.base);
        };
        for (const plan::Schema& s : C.schemas)
        {
            mark(s.pre[0]);
            mark(s.pre[1]);
            for (const plan::CondEffect& ce : s.ces)
                mark(ce.cond);
        }
        for (const plan::Stratum& st : C.strata)
            for (const plan::Axiom& x : st.axioms)
                mark(x.body);
        m_view_map.assign(R, dev::k_none);
        m_view_live.assign(R, 0);
        m_view_rows = 0;
        bool zero = false;
        for (u32 i = 0; i < R; ++i)
            if (read[i] && live[i])
            {
                m_view_map[i] = m_view_rows++;
                m_view_live[i] = 1;
            }
            else if (read[i])
                zero = true;
        if (zero)
        {
            for (u32 i = 0; i < R; ++i)
                if (read[i] && !live[i])
                    m_view_map[i] = m_view_rows;
            ++m_view_rows;
        }
    }

    static bool fixed_in_order(const plan::Matcher& m)
    {
        for (u32 d = 0; d < m.steps.size(); ++d)
            if (m.steps[d].param != m.first_free + d)
                return false;
        return true;
    }
    static void put(u32* row, u32 col, std::pair<u32, u32> r)
    {
        row[col] = r.first;
        row[col + 1] = r.second;
    }

    u32 pattern(const plan::Pattern& pt)
    {
        const u32 id = count(m_pattern_base.size());
        m_pattern_rows.insert(m_pattern_rows.end(), {static_cast<u32>(pt.kind), pt.pred, pt.var_begin, pt.var_count});
        m_pattern_base.push_back(pt.base);
        const bool is_static = pt.kind == plan::LitKind::Static;
        const std::vector<u64>& tabs = is_static ? C.statics[pt.pred].rs : C.layout.rs;
        const u64 base = is_static ? m_static_rs[pt.pred] : 0;
        for (u32 i = 0; i < pt.var_count; ++i)
        {
            const plan::PatternVar& var = C.pattern_vars[pt.var_begin + i];
            const std::ptrdiff_t off = var.rs - tabs.data();
            if (off < 0 || static_cast<u64>(off) + C.num_objects > tabs.size())
                throw std::logic_error("mymyr: device_arrays: a pattern variable points outside its position tables");
            m_vars[(pt.var_begin + i) * usize{2}] = var.param;
            m_vars[(pt.var_begin + i) * usize{2} + 1] = count(base + static_cast<u64>(off));
        }
        return id;
    }
    std::pair<u32, u32> patterns(const std::vector<plan::Pattern>& v)
    {
        const u32 first = count(m_pattern_base.size());
        for (const plan::Pattern& pt : v)
            (void)pattern(pt);
        return {first, count(v.size())};
    }
    std::pair<u32, u32> checks(const std::vector<plan::Check>& v)
    {
        std::vector<u32> ids;
        ids.reserve(v.size());
        for (const plan::Check& c : v)
            ids.push_back(pattern(c.pat));
        const u32 first = count(m_check_rows.size() / 2);
        for (usize i = 0; i < v.size(); ++i)
            m_check_rows.insert(m_check_rows.end(), {ids[i], v[i].pos ? 1u : 0u});
        return {first, count(v.size())};
    }
    template<class T>
    std::pair<u32, u32> list(const std::vector<T>& v)
    {
        const u32 first = count(m_index.size());
        for (const T& x : v)
            m_index.push_back(static_cast<u32>(x));
        return {first, count(v.size())};
    }
    std::pair<u32, u32> row_list(const std::vector<plan::Row>& v)
    {
        const u32 first = count(m_rows.size() / 3);
        for (const plan::Row& r : v)
            m_rows.insert(m_rows.end(), {r.base.v, u64{r.src}, r.neg ? u64{1} : u64{0}});
        return {first, count(v.size())};
    }

    /// A matcher row; `fc`: the device prefers forward checking (rl::device_search_costs).
    u32 matcher(const plan::Matcher& m, bool fc)
    {
        std::array<u32, dev::k_mc_count> row{};
        row[dev::k_mc_total] = m.total;
        row[dev::k_mc_first_free] = m.first_free;
        row[dev::k_mc_flags] = (m.never ? dev::k_mc_never : 0u) | (m.use_fc ? dev::k_mc_use_fc : 0u) |
                               (m.binds_in_order ? dev::k_mc_binds_in_order : 0u) |
                               ((C.num.slots ? m.use_fc : fc) && C.ow <= dev::k_max_fc_ow ? dev::k_mc_device_fc : 0u) |
                               (fixed_in_order(m) ? dev::k_mc_fixed_in_order : 0u) |
                               (m.steps.size() >= m.first_exist + 2 ? dev::k_mc_witnesses : 0u);
        row[dev::k_mc_first_exist] = m.first_exist;
        row[dev::k_mc_dom0] = count(m_dom0.size());
        row[dev::k_mc_dom0_n] = count(m.dom0.size());
        m_dom0.insert(m_dom0.end(), m.dom0.begin(), m.dom0.end());
        put(row.data(), dev::k_mc_unary_begin, list(m.unary_begin));
        row[dev::k_mc_unary] = count(m_unary.size() / 2);
        row[dev::k_mc_unary_n] = count(m.unary.size());
        for (const plan::Unary& u : m.unary)
            m_unary.insert(m_unary.end(), {u.ptr.v, u.neg ? u64{1} : u64{0}});
        put(row.data(), dev::k_mc_pre_checks, checks(m.pre_checks));
        put(row.data(), dev::k_mc_checks, checks(m.checks));
        row[dev::k_mc_steps] = count(m_steps.size() / 5);
        row[dev::k_mc_steps_n] = count(m.steps.size());
        for (const plan::Step& st : m.steps)
            m_steps.insert(m_steps.end(), {st.param, st.row_begin, st.row_end, st.check_begin, st.check_end});
        put(row.data(), dev::k_mc_rows, row_list(m.rows));
        put(row.data(), dev::k_mc_step_checks, list(m.step_checks));
        put(row.data(), dev::k_mc_free_params, list(m.free_params));
        put(row.data(), dev::k_mc_relevant, list(m.relevant));
        put(row.data(), dev::k_mc_fc_out_begin, list(m.fc_out_begin));
        row[dev::k_mc_fc_out] = count(m_edges.size() / 3);
        row[dev::k_mc_fc_out_n] = count(m.fc_out.size());
        for (const plan::Edge& e : m.fc_out)
            m_edges.insert(m_edges.end(), {u64{e.to}, e.base.v, e.neg ? u64{1} : u64{0}});
        put(row.data(), dev::k_mc_fc_pre, row_list(m.fc_pre));
        put(row.data(), dev::k_mc_fc_pre_to, list(m.fc_pre_to));
        put(row.data(), dev::k_mc_fc_checks_begin, list(m.fc_checks_begin));
        put(row.data(), dev::k_mc_fc_checks, list(m.fc_checks));
        put(row.data(), dev::k_mc_check_vars_begin, list(m.check_vars_begin));
        put(row.data(), dev::k_mc_check_vars, list(m.check_vars));
        const u32 id = count(m_matcher_rows.size() / dev::k_mc_count);
        m_matcher_rows.insert(m_matcher_rows.end(), row.begin(), row.end());
        m_numeric_matchers.push_back(&m);
        return id;
    }

    const Task& m_task;
    const plan::Compiled& C;
    u32 m_fluent, m_derived;
    DeviceSearchCosts m_costs;  // the device's matchers (k_mc_device_fc)
    std::vector<u64> m_static_rs;
    std::vector<u32> m_view_map;  // ViewLayout row -> device view row (k_none: no matcher reads it)
    std::vector<u8> m_view_live;  // ViewLayout row -> kept (not the zero row, not dropped)
    u32 m_view_rows = 0;
    std::vector<u64> m_rs, m_pattern_base, m_dom0, m_unary, m_rows, m_edges;
    std::vector<const plan::Matcher*> m_numeric_matchers;
    std::vector<u32> m_vars, m_pattern_rows, m_check_rows, m_matcher_rows, m_index, m_steps, m_cond_effect;
};

void PlanWriter::write_numeric(ArrayBundle::Builder& b, const std::vector<u32>& schema)
{
    const plan::Numeric& n = C.num;
    std::vector<u32> code, checks, matchers, steps, indices, masks, effects, groups, schemas, order, ces;
    std::vector<u64> tables, rs, payload;
    for (const plan::NumIns& in : n.code)
        code.insert(code.end(), {static_cast<u32>(in.op), in.ar, in.a, in.b});
    for (const plan::FunctionTable& t : n.tables)
    {
        const u64 ro = rs.size();
        rs.insert(rs.end(), t.rs.begin(), t.rs.end());
        const u64 dense = payload.size();
        if (t.fluent)
            payload.insert(payload.end(), t.slot.begin(), t.slot.end());
        else
            for (f64 v : t.value)
                payload.push_back(std::bit_cast<u64>(v));
        const u64 dn = payload.size() - dense;
        const u64 keys = t.hkeys.empty() ? ~u64{0} : payload.size();
        payload.insert(payload.end(), t.hkeys.begin(), t.hkeys.end());
        const u64 values = payload.size();
        if (t.fluent)
            payload.insert(payload.end(), t.hslot.begin(), t.hslot.end());
        else
            for (f64 v : t.hvalue)
                payload.push_back(std::bit_cast<u64>(v));
        tables.insert(tables.end(), {t.size, ro, dense, dn, keys, values, t.hmask, t.fluent ? 1u : 0u});
    }
    auto put_checks = [&](const std::vector<plan::NumCheck>& cs)
    {
        const u32 begin = count(checks.size() / 5);
        for (const plan::NumCheck& c : cs)
            checks.insert(checks.end(), {static_cast<u32>(c.cmp), c.lhs.begin, c.lhs.end, c.rhs.begin, c.rhs.end});
        return begin;
    };
    for (const plan::Matcher* pm : m_numeric_matchers)
    {
        const plan::Matcher& m = *pm;
        const u32 pre = put_checks(m.npre), check = put_checks(m.nchecks);
        const u32 at = count(steps.size() / 2), mask = count(masks.size());
        for (const plan::Step& st : m.steps)
        {
            const u32 begin = count(indices.size());
            for (u32 j = st.ncheck_begin; j < st.ncheck_end; ++j)
                indices.push_back(check + m.step_nchecks[j]);
            steps.insert(steps.end(), {begin, count(indices.size())});
        }
        for (u32 i = 0; i < m.nchecks.size(); ++i)
        {
            u32 bits = 0;
            for (u32 j = m.ncheck_vars_begin[i]; j < m.ncheck_vars_begin[i + 1]; ++j)
            {
                if (m.ncheck_vars[j] < 32)
                    bits |= u32{1} << m.ncheck_vars[j];
            }
            masks.push_back(bits);
        }
        matchers.insert(matchers.end(), {pre, count(m.npre.size()), check, count(m.nchecks.size()), at, mask, 0, 0});
    }
    auto group = [&](const auto& g)
    {
        const u32 id = count(groups.size() / 5), begin = count(effects.size() / 8);
        for (const plan::NumEffect& e : g.neffs)
            effects.insert(effects.end(), {static_cast<u32>(e.op), e.slot, e.lifted ? 1u : 0u, e.table, e.terms,
                                          e.arity, e.expr.begin, e.expr.end});
        groups.insert(groups.end(), {begin, count(g.neffs.size()), g.has_aux ? static_cast<u32>(g.aux.op) : dev::k_none,
                                      g.aux.expr.begin, g.aux.expr.end});
        return id;
    };
    for (u32 s = 0; s < C.schemas.size(); ++s)
    {
        const plan::Schema& ps = C.schemas[s];
        std::vector<u32> ug;
        for (const plan::NumGroup& g : ps.uncond_num)
            ug.push_back(group(g));
        for (const plan::CondEffect& ce : ps.ces)
            ces.insert(ces.end(), {group(ce), ce.numeric ? 1u : 0u, ce.extras ? 1u : 0u});
        schemas.insert(schemas.end(), {count(order.size() / 2), count(ps.num_order.size())});
        for (const plan::NumRef& r : ps.num_order)
            order.insert(order.end(), {r.conditional ? 1u : 0u,
                                      r.conditional ? schema[u64{s} * dev::k_sc_count + dev::k_sc_ces] + r.index : ug[r.index]});
    }
    const u32 goal = put_checks(n.goal);
    b.add("num_code", DType::U32, shape2(code.size() / 4, 4), code);
    b.add("num_consts", DType::F64, shape1(n.consts.size()), n.consts);
    b.add("num_terms", DType::I32, shape1(n.terms.size()), n.terms);
    b.add("num_tables", DType::U64, shape2(tables.size() / 8, 8), tables);
    b.add("num_rs", DType::U64, shape1(rs.size()), rs);
    b.add("num_payload", DType::U64, shape1(payload.size()), payload);
    b.add("num_checks", DType::U32, shape2(checks.size() / 5, 5), checks);
    b.add("num_matchers", DType::U32, shape2(matchers.size() / 8, 8), matchers);
    b.add("num_steps", DType::U32, shape2(steps.size() / 2, 2), steps);
    b.add("num_index", DType::U32, shape1(indices.size()), indices);
    b.add("num_masks", DType::U32, shape1(masks.size()), masks);
    b.add("num_effects", DType::U32, shape2(effects.size() / 8, 8), effects);
    b.add("num_groups", DType::U32, shape2(groups.size() / 5, 5), groups);
    b.add("num_schema", DType::U32, shape2(schemas.size() / 2, 2), schemas);
    b.add("num_order", DType::U32, shape2(order.size() / 2, 2), order);
    b.add("num_ce", DType::U32, shape2(ces.size() / 3, 3), ces);
    std::vector<f64> options{n.quantum, n.aux_initial};
    b.add("num_options", DType::F64, shape1(options.size()), options);
    b.add("num_initial", DType::F64, shape1(n.initial.size()), n.initial);
    b.scalar("section_numeric", 1);
    b.scalar("num_has_aux", n.has_aux ? 1 : 0);
    b.scalar("num_has_metric", n.has_metric ? 1 : 0);
    b.scalar("num_minimize", n.minimize ? 1 : 0);
    b.scalar("num_metric_begin", n.metric.begin);
    b.scalar("num_metric_end", n.metric.end);
    b.scalar("num_tolerant", n.tolerant ? 1 : 0);
    b.scalar("num_goal_begin", goal);
    b.scalar("num_goal_count", n.goal.size());
}

void add_plan_section(ArrayBundle::Builder& b, const Task& task, u32 fluent_slots, u32 derived_slots)
{
    PlanWriter(task, fluent_slots, derived_slots).write(b);
}
}  // namespace

ArrayBundle atom_metadata(const Task& task)
{
    ArrayBundle::Builder b;
    const Slots fluent = published(task, AtomKind::Fluent);
    const Slots derived = published(task, AtomKind::Derived);
    add_metadata(b, task, fluent, derived);
    return b.build();
}

GoalMasks goal_masks(const Task& task)
{
    GoalMasks g;
    const plan::Goal& goal = task.compiled().goal;
    g.unsatisfiable = goal.unsatisfiable;
    const AtomIndex& A = task.atoms();
    u32 max_slot = 0;
    std::vector<std::pair<u32, bool>> fluent;
    for (const plan::Check& c : goal.lits)
    {
        // ground patterns: the key is the base (a canonical id). An atom outside the layout is in no state (no effect
        // or axiom produces it and the initial state lacks it): as a positive goal it is unreachable, as a negative
        // one it always holds.
        if (c.pat.base >= A.layout().total)
        {
            g.unsatisfiable |= c.pos;
            continue;
        }
        const u32 s = A.intern(c.pat.base);
        if (c.pat.kind == plan::LitKind::Derived)
            (c.pos ? g.derived_pos : g.derived_neg).push_back(s);
        else
        {
            fluent.emplace_back(s, c.pos);
            max_slot = std::max(max_slot, s + 1);
        }
    }
    const u32 W = bits::words_for(max_slot);
    g.pos.assign(W, 0);
    g.neg.assign(W, 0);
    for (auto [s, pos] : fluent)
        bits::set((pos ? g.pos : g.neg).data(), s);
    return g;
}

ArrayBundle device_arrays(const Task& task, u32 version)
{
    if (version != 1 && version != 2)
        throw std::invalid_argument("mymyr: device_arrays: unknown version " + std::to_string(version) +
                                    " (this build provides version " + std::to_string(k_device_arrays_version) + ")");
    const GoalMasks goal = goal_masks(task);  // first: it may assign slots
    ArrayBundle::Builder b;
    const Slots fluent = published(task, AtomKind::Fluent);
    const Slots derived = published(task, AtomKind::Derived);
    add_metadata(b, task, fluent, derived);

    const plan::Compiled& C = task.compiled();
    const formalism::TaskData& D = task.data();
    const State init = task.initial_state();
    const u32 W = std::max<u32>({bits::words_for(fluent.count), init.size_words(), static_cast<u32>(goal.pos.size()),
                                 static_cast<u32>(goal.neg.size())});
    b.add("init", DType::U64, shape1(W), padded_words(init.words(), W), true);
    b.add("goal_pos", DType::U64, shape1(W), padded_words(goal.pos, W), true);
    b.add("goal_neg", DType::U64, shape1(W), padded_words(goal.neg, W), true);
    std::vector<i32> gdp(goal.derived_pos.begin(), goal.derived_pos.end()), gdn(goal.derived_neg.begin(), goal.derived_neg.end());
    b.add("goal_derived_pos", DType::I32, shape1(gdp.size()), gdp);
    b.add("goal_derived_neg", DType::I32, shape1(gdn.size()), gdn);

    // typed-dense canonical layout: cid(p, a) = pred_offset[p] + sum_i pos_rank[pos_begin[p] + i][a_i]
    // * pos_stride[pos_begin[p] + i]; a rank of -1 means the object never occurs at that position
    const CanonicalLayout& L = C.layout;
    const u32 P = static_cast<u32>(L.arity.size());
    const u64 npos = L.stride.size();
    const u32 n = L.num_objects;
    std::vector<i32> poff(P, -1), psize(P, 0), pbegin(P, -1), pstride(npos), prank(npos * n, -1);
    for (u32 p = 0; p < P; ++p)
        if (L.has_predicate(p))
        {
            poff[p] = static_cast<i32>(L.offset[p]);
            psize[p] = static_cast<i32>(L.size[p]);
            pbegin[p] = static_cast<i32>(L.pos_begin[p]);
        }
    for (u64 q = 0; q < npos; ++q)
    {
        pstride[q] = static_cast<i32>(L.stride[q]);
        for (u32 o = 0; o < n; ++o)
        {
            const u64 rs = L.rs[q * n + o];
            if (rs < CanonicalLayout::k_outside && L.stride[q])
                prank[q * n + o] = static_cast<i32>(rs / L.stride[q]);
        }
    }
    b.add("pred_offset", DType::I32, shape1(P), poff);
    b.add("pred_size", DType::I32, shape1(P), psize);
    b.add("pos_begin", DType::I32, shape1(P), pbegin);
    b.add("pos_stride", DType::I32, shape1(npos), pstride);
    b.add("pos_rank", DType::I32, shape2(npos, n), prank);

    const u32 S = static_cast<u32>(D.schemas.size());
    std::vector<i32> sar(S), sor(S);
    for (u32 s = 0; s < S; ++s)
    {
        sar[s] = static_cast<i32>(D.schemas[s].arity());
        sor[s] = static_cast<i32>(D.schemas[s].original_arity);
    }
    b.add("schema_arity", DType::I32, shape1(S), sar);
    b.add("schema_original_arity", DType::I32, shape1(S), sor);

    // numeric tasks: the initial state's numeric words ([bits | slots] rows: they follow the atom words) and
    // their encoding (numeric_storage 1: two int32 per word, 0: one double per word). Classical tasks: absent (v1).
    if (task.numeric_slots() > 0)
    {
        const std::span<const u64> num = init.numeric();
        b.add("init_numeric", DType::U64, shape1(num.size()), std::vector<u64>(num.begin(), num.end()), true);
        b.scalar("numeric_slots", task.numeric_slots());
        b.scalar("numeric_words", task.numeric_words());
        b.scalar("numeric_storage", task.numeric_storage() == NumericStorage::I32 ? 1 : 0);
    }

    b.scalar("version", version);
    b.scalar("num_schemas", S);
    b.scalar("label_width", max_label_width(task));
    b.scalar("object_words", C.ow);
    b.scalar("state_words", W);
    b.scalar("goal_uses_derived", goal.uses_derived() ? 1 : 0);
    b.scalar("goal_unsatisfiable", goal.unsatisfiable ? 1 : 0);
    if (version >= 2)
    {
        b.scalar("section_core", k_section_core_version);
        add_plan_section(b, task, fluent.count, derived.count);
    }
    return b.build();
}

dev::TaskView task_view(const ArrayBundle& bundle, const void* base)
{
    if (bundle.scalar("version", 0) < 2 || bundle.scalar("section_plan", 0) != k_section_plan_version)
        throw std::invalid_argument("mymyr: task_view: needs a device_arrays export of version 2 or later with section "
                                    "plan " + std::to_string(k_section_plan_version));
    const auto* block = base ? static_cast<const std::byte*>(base) : bundle.block();
    auto ptr = [&]<class T>(const char* name, const T*& out) -> u64
    {
        const ArrayInfo* a = bundle.find(name);
        if (!a)
            throw std::invalid_argument(std::string("mymyr: task_view: the export has no array '") + name + "'");
        out = reinterpret_cast<const T*>(block + a->offset);
        return a->shape.empty() ? 1 : static_cast<u64>(a->shape[0]);
    };
    auto u32s = [&](const char* name) { return static_cast<u32>(bundle.scalar(name, 0)); };
    dev::TaskView v;
    v.num_objects = u32s("num_objects");
    v.ow = u32s("object_words");
    v.state_words = u32s("state_words");
    v.num_preds = u32s("num_predicates");
    v.num_schemas = u32s("num_schemas");
    v.max_bind = u32s("plan_max_bind");
    v.view_rows = u32s("plan_view_rows");
    v.fluent_slots = u32s("plan_fluent_slots");
    v.derived_slots = u32s("plan_derived_slots");
    v.atom_total = static_cast<u64>(bundle.scalar("plan_atom_total", 0));
    v.fluent_total = static_cast<u64>(bundle.scalar("plan_fluent_total", 0));
    v.goal_check = u32s("plan_goal_check");
    v.goal_check_n = u32s("plan_goal_check_n");
    v.goal_unsatisfiable = u32s("goal_unsatisfiable");
    ptr("init", v.init);
    ptr("goal_pos", v.goal_pos);
    ptr("goal_neg", v.goal_neg);
    ptr("plan_slot_of", v.slot_of);
    v.n_rs = ptr("plan_rs", v.rs);
    v.n_static_words = ptr("plan_static_words", v.static_words);
    ptr("plan_static_rel", v.static_rel);
    v.n_static_bits = ptr("plan_static_bits", v.static_bits);
    v.n_static_table = ptr("plan_static_table", v.static_table);
    ptr("plan_fluent_view_op", v.fluent_view_op);
    ptr("plan_derived_view_op", v.derived_view_op);
    v.n_view_map = static_cast<u32>(ptr("plan_view_map", v.view_map));
    v.n_patterns = static_cast<u32>(ptr("plan_pattern", v.pattern));
    ptr("plan_pattern_base", v.pattern_base);
    v.n_pattern_vars = static_cast<u32>(ptr("plan_pattern_var", v.pattern_var));
    v.n_checks = static_cast<u32>(ptr("plan_check", v.check));
    v.n_matchers = static_cast<u32>(ptr("plan_matcher", v.matcher));
    v.n_dom0 = ptr("plan_dom0", v.dom0);
    v.n_index = ptr("plan_index", v.index);
    v.n_unary = static_cast<u32>(ptr("plan_unary", v.unary));
    v.n_rows = static_cast<u32>(ptr("plan_row", v.row));
    v.n_edges = static_cast<u32>(ptr("plan_edge", v.edge));
    v.n_steps = static_cast<u32>(ptr("plan_step", v.step));
    ptr("plan_schema", v.schema);
    v.n_cond_effects = static_cast<u32>(ptr("plan_cond_effect", v.cond_effect));
    v.n_axioms = static_cast<u32>(ptr("plan_axiom", v.axiom));
    v.n_strata = static_cast<u32>(ptr("plan_stratum", v.stratum));
    if (u32s("numeric_slots"))
    {
        if (u32s("section_numeric") != 1)
            throw std::invalid_argument("mymyr: task_view: numeric tasks need numeric section 1");
        auto& n = v.numeric;
        n.slots = u32s("numeric_slots");
        n.storage = u32s("numeric_storage");
        n.tolerant = u32s("num_tolerant");
        n.goal_begin = u32s("num_goal_begin");
        n.goal_count = u32s("num_goal_count");
        const f64* options = nullptr;
        ptr("num_options", options);
        n.quantum = *reinterpret_cast<const f64*>(bundle.block() + bundle.find("num_options")->offset);
        ptr("num_code", n.code);
        ptr("num_consts", n.consts);
        ptr("num_terms", n.terms);
        ptr("num_tables", n.table);
        ptr("num_rs", n.rs);
        ptr("num_payload", n.payload);
        ptr("num_checks", n.checks);
        ptr("num_matchers", n.matcher);
        ptr("num_steps", n.steps);
        ptr("num_index", n.index);
        ptr("num_masks", n.masks);
        ptr("num_effects", n.effects);
        ptr("num_groups", n.groups);
        ptr("num_schema", n.schema);
        ptr("num_order", n.order);
        ptr("num_ce", n.ce);
        ptr("num_initial", n.initial);
    }
    return v;
}
}  // namespace mymyr::rl
