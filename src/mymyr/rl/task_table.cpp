// TaskTable: many instances of one domain (include/mymyr/rl/task_table.hpp).

#include "mymyr/rl/task_table.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/core/hash.hpp"
#include "mymyr/rl/expand.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace mymyr::rl
{
namespace
{
i32 kind_code(formalism::PredKind k)
{
    return k == formalism::PredKind::Static ? k_pred_static : k == formalism::PredKind::Fluent ? k_pred_fluent : k_pred_derived;
}

std::string kind_text(i32 k) { return k == k_pred_static ? "static" : k == k_pred_fluent ? "fluent" : "derived"; }

/// Per predicate: 1 if it is problem-local, i.e. derived and defined by problem axioms only (goal normalization's
/// "axiom_<k>"), else 0.
std::vector<u8> problem_local(const formalism::TaskData& D)
{
    std::vector<u8> any(D.predicates.size(), 0), domain(D.predicates.size(), 0);
    for (const formalism::Axiom& x : D.axioms)
    {
        any[x.head.pred.v] = 1;
        if (!x.from_problem)
            domain[x.head.pred.v] = 1;
    }
    std::vector<u8> out(D.predicates.size(), 0);
    for (usize p = 0; p < D.predicates.size(); ++p)
        out[p] = D.predicates[p].kind == formalism::PredKind::Derived && any[p] && !domain[p] ? 1 : 0;
    return out;
}

[[noreturn]] void not_one_domain(u32 i, const std::string& why)
{
    throw std::invalid_argument("mymyr: TaskTable: instance " + std::to_string(i) +
                                " is not of the domain of instance 0: " + why);
}

/// A task's domain predicates (every predicate but the problem-local ones), in its order.
struct DomainPred
{
    std::string name;
    u32 arity;
    i32 kind;
};
std::vector<DomainPred> domain_predicates(const formalism::TaskData& D)
{
    const std::vector<u8> local = problem_local(D);
    std::vector<DomainPred> out;
    for (usize p = 0; p < D.predicates.size(); ++p)
        if (!local[p])
            out.push_back({std::string(D.str(D.predicates[p].name)), D.predicates[p].arity, kind_code(D.predicates[p].kind)});
    return out;
}
}  // namespace

std::string TaskTable::domain_mismatch(const Task& ref, const Task& task)
{
    const formalism::TaskData& R = ref.data();
    const formalism::TaskData& D = task.data();
    if (D.schemas.size() != R.schemas.size())
        return std::to_string(D.schemas.size()) + " action schemas, instance 0 has " + std::to_string(R.schemas.size());
    for (usize s = 0; s < D.schemas.size(); ++s)
        if (D.str(D.schemas[s].name) != R.str(R.schemas[s].name) || D.schemas[s].arity() != R.schemas[s].arity())
            return "action schema " + std::to_string(s) + " is " + std::string(D.str(D.schemas[s].name)) + "/" +
                   std::to_string(D.schemas[s].arity()) + ", instance 0 has " + std::string(R.str(R.schemas[s].name)) +
                   "/" + std::to_string(R.schemas[s].arity());
    const std::vector<DomainPred> rp = domain_predicates(R), dp = domain_predicates(D);
    for (usize d = 0; d < dp.size(); ++d)
        if (d >= rp.size() || rp[d].name != dp[d].name || rp[d].arity != dp[d].arity || rp[d].kind != dp[d].kind)
            return "domain predicate " + std::to_string(d) + " is " + dp[d].name + "/" + std::to_string(dp[d].arity) +
                   " (" + kind_text(dp[d].kind) + "), instance 0 has " +
                   (d < rp.size() ? rp[d].name + "/" + std::to_string(rp[d].arity) + " (" + kind_text(rp[d].kind) + ")"
                                  : std::string("no more"));
    if (dp.size() != rp.size())
        return std::to_string(dp.size()) + " domain predicates, instance 0 has " + std::to_string(rp.size());
    return {};
}

TaskTablePtr TaskTable::create(std::vector<TaskPtr> tasks)
{
    return std::make_shared<const TaskTable>(Private{}, std::move(tasks));
}

TaskTablePtr TaskTable::single(TaskPtr task)
{
    std::vector<TaskPtr> v;
    v.push_back(std::move(task));
    return create(std::move(v));
}

TaskTable::TaskTable(Private, std::vector<TaskPtr> tasks)
{
    if (tasks.empty())
        throw std::invalid_argument("mymyr: TaskTable: no instances");
    if (tasks.size() > static_cast<usize>(std::numeric_limits<i32>::max()))
        throw std::invalid_argument("mymyr: TaskTable: more than 2^31 - 1 instances");
    for (usize i = 0; i < tasks.size(); ++i)
        if (!tasks[i])
            throw std::invalid_argument("mymyr: TaskTable: instance " + std::to_string(i) + " is null");

    // schemas: instance 0's, checked against every other instance (count, names, arities, order)
    {
        const formalism::TaskData& D = tasks[0]->data();
        for (const formalism::Schema& s : D.schemas)
        {
            m_schema_names.emplace_back(D.str(s.name));
            m_schema_arity.push_back(s.arity());
            m_label_width = std::max(m_label_width, s.arity());
        }
    }
    // predicates: the domain's (every instance's, in the same order: their table ids are the domain's own) first, then
    // the problem-local ones (derived predicates defined by problem axioms only, e.g. goal normalization's "axiom_0"),
    // one table predicate per (instance, predicate): two problems' "axiom_0" are unrelated
    struct Local
    {
        u32 instance, pred;
    };
    std::vector<Local> locals;
    for (usize i = 0; i < tasks.size(); ++i)
    {
        const Task& T = *tasks[i];
        const formalism::TaskData& D = T.data();
        const u32 ii = static_cast<u32>(i);
        if (i > 0)
            if (const std::string why = domain_mismatch(*tasks[0], T); !why.empty())
                not_one_domain(ii, why);
        const std::vector<u8> local = problem_local(D);
        Instance in;
        in.task = tasks[i];
        in.pred_map.resize(D.predicates.size());
        u32 d = 0;  // the next domain predicate
        for (usize p = 0; p < D.predicates.size(); ++p)
        {
            const formalism::Predicate& P = D.predicates[p];
            if (local[p])
            {
                locals.push_back({ii, static_cast<u32>(p)});
                continue;
            }
            if (i == 0)
            {
                m_pred_names.emplace_back(D.str(P.name));
                m_pred_arity.push_back(P.arity);
                m_pred_kind.push_back(kind_code(P.kind));
                m_pred_instance.push_back(-1);
            }
            in.pred_map[p] = d++;  // domain_mismatch: instance i's domain predicates are instance 0's, in its order
        }
        if (i == 0)
            m_domain_preds = d;
        in.num_objects = T.num_objects();
        in.words = std::max<u32>(1, T.max_words());
        in.numeric_words = T.numeric_words();
        in.ow = T.compiled().ow;
        const GoalMasks g = goal_masks(T);  // assigns the goal atoms' slots under lazy slots (as device_arrays does)
        in.goal_pos = g.pos;
        in.goal_neg = g.neg;
        in.goal_derived = g.uses_derived();
        in.goal_unsatisfiable = g.unsatisfiable;
        m_words = std::max(m_words, in.words);
        m_numeric_words = std::max(m_numeric_words, in.numeric_words);
        m_max_objects = std::max(m_max_objects, in.num_objects);
        m_numeric = m_numeric || T.numeric_slots() > 0;
        m_inst.push_back(std::move(in));
    }
    // the problem-local predicates after the domain's, in (instance, predicate) order
    for (const Local& l : locals)
    {
        const formalism::TaskData& D = m_inst[l.instance].task->data();
        const formalism::Predicate& P = D.predicates[l.pred];
        m_inst[l.instance].pred_map[l.pred] = static_cast<u32>(m_pred_names.size());
        m_pred_names.emplace_back(D.str(P.name));
        m_pred_arity.push_back(P.arity);
        m_pred_kind.push_back(kind_code(P.kind));
        m_pred_instance.push_back(static_cast<i32>(l.instance));
    }
    // rows at the table width
    for (Instance& in : m_inst)
    {
        const Task& T = *in.task;
        in.init.assign(static_cast<usize>(m_words) + m_numeric_words, 0);
        const State s = T.initial_state();
        std::copy_n(s.data(), std::min(s.size_words(), m_words), in.init.begin());
        const auto num = s.numeric();
        std::copy(num.begin(), num.end(), in.init.begin() + m_words);
        in.goal_pos.resize(m_words, 0);
        in.goal_neg.resize(m_words, 0);
    }
}

u64 TaskTable::fingerprint() const
{
    u64 h = hash::k_seed ^ m_inst.size();
    for (const Instance& in : m_inst)
        h = hash::combine(h, in.task->fingerprint());
    return h;
}

void TaskTable::check_task_ids(const i32* task_ids, u64 n) const
{
    if (!task_ids)
        return;
    const i64 I = static_cast<i64>(m_inst.size());
    for (u64 r = 0; r < n; ++r)
        if (task_ids[r] < 0 || task_ids[r] >= I)
            throw std::invalid_argument("mymyr: task id " + std::to_string(task_ids[r]) + " of row " + std::to_string(r) +
                                        " is outside the table's " + std::to_string(I) + " instances");
}

ArrayBundle table_atom_metadata(const TaskTable& table)
{
    const u32 I = table.size();
    ArrayBundle::Builder b;
    struct Part
    {
        std::vector<i32> pred, args, cid;
        std::vector<i64> offsets{0};
        u32 width = 0;
    };
    Part fluent, derived, statics;
    std::vector<i32> num_objects(I), num_atoms(I), words(I), domain(I, 0);
    std::vector<i64> object_offsets(I + 1, 0);
    std::vector<ArrayBundle> meta;
    meta.reserve(I);
    for (u32 i = 0; i < I; ++i)
    {
        meta.push_back(atom_metadata(*table.task(i)));
        for (const char* prefix : {"atom", "derived", "static"})
        {
            const ArrayInfo* a = meta.back().find(std::string(prefix) + "_args_padded");
            Part& part = prefix[0] == 'a' ? fluent : prefix[0] == 'd' ? derived : statics;
            part.width = std::max<u32>(part.width, a && a->shape.size() == 2 ? static_cast<u32>(a->shape[1]) : 0);
        }
    }
    for (u32 i = 0; i < I; ++i)
    {
        const ArrayBundle& m = meta[i];
        const TaskTable::Instance& in = table.instance(i);
        auto append = [&](const std::string& prefix, Part& part, bool cid)
        {
            const ArrayInfo* pa = m.find(prefix + "_pred");
            const ArrayInfo* aa = m.find(prefix + "_args_padded");
            const u64 n = pa ? pa->elements() : 0;
            const auto* pred = pa ? static_cast<const i32*>(m.data(*pa)) : nullptr;
            const auto* args = aa ? static_cast<const i32*>(m.data(*aa)) : nullptr;
            const u32 A = aa && aa->shape.size() == 2 ? static_cast<u32>(aa->shape[1]) : 0;
            for (u64 j = 0; j < n; ++j)
            {
                part.pred.push_back(static_cast<i32>(in.pred_map[static_cast<u32>(pred[j])]));
                for (u32 k = 0; k < part.width; ++k)
                    part.args.push_back(k < A ? args[j * A + k] : -1);
            }
            if (cid)
            {
                const ArrayInfo* ca = m.find(prefix + "_cid");
                const auto* c = static_cast<const i32*>(m.data(*ca));
                part.cid.insert(part.cid.end(), c, c + n);
            }
            part.offsets.push_back(part.offsets.back() + static_cast<i64>(n));
        };
        append("atom", fluent, true);
        append("derived", derived, true);
        append("static", statics, false);
        num_objects[i] = static_cast<i32>(in.num_objects);
        object_offsets[i + 1] = object_offsets[i] + in.num_objects;
        num_atoms[i] = static_cast<i32>(fluent.offsets[i + 1] - fluent.offsets[i]);
        words[i] = static_cast<i32>(bits::words_for(static_cast<u64>(num_atoms[i])));
    }
    auto shape1 = [](u64 n) { return std::vector<i64>{static_cast<i64>(n)}; };
    auto shape2 = [](u64 n, u64 m) { return std::vector<i64>{static_cast<i64>(n), static_cast<i64>(m)}; };
    auto add_part = [&](const std::string& prefix, const Part& p, bool cid)
    {
        const u64 n = p.pred.size();
        b.add(prefix + "_offsets", DType::I64, shape1(p.offsets.size()), p.offsets);
        b.add(prefix + "_pred", DType::I32, shape1(n), p.pred);
        b.add(prefix + "_args", DType::I32, shape2(n, p.width), p.args);
        if (cid)
            b.add(prefix + "_cid", DType::I32, shape1(n), p.cid);
    };
    add_part("atom", fluent, true);
    add_part("derived", derived, true);
    add_part("static", statics, false);
    b.add("num_objects", DType::I32, shape1(I), num_objects);
    b.add("object_offsets", DType::I64, shape1(I + 1), object_offsets);
    b.add("num_atoms", DType::I32, shape1(I), num_atoms);
    b.add("words", DType::I32, shape1(I), words);
    b.add("domain", DType::I32, shape1(I), domain);
    std::vector<i32> local(I);
    std::iota(local.begin(), local.end(), 0);
    b.add("local_id", DType::I32, shape1(I), local);
    const u32 P = static_cast<u32>(table.predicate_names().size());
    std::vector<i32> parity(P), pkind(P);
    u32 max_arity = 0;
    for (u32 p = 0; p < P; ++p)
    {
        parity[p] = static_cast<i32>(table.predicate_arities()[p]);
        pkind[p] = table.predicate_kinds()[p];
        max_arity = std::max(max_arity, table.predicate_arities()[p]);
    }
    // the per-domain fields of a suite's metadata (task_suite.hpp), for the table's one domain
    b.add("pred_offsets", DType::I64, shape1(2), std::vector<i64>{0, P});
    b.add("pred_arity", DType::I32, shape1(P), parity);
    b.add("pred_kind", DType::I32, shape1(P), pkind);
    b.add("pred_instance", DType::I32, shape1(P), table.predicate_instances());
    b.add("domain_predicates", DType::I32, shape1(1), std::vector<i32>{static_cast<i32>(table.num_domain_predicates())});
    b.add("schema_offsets", DType::I64, shape1(2), std::vector<i64>{0, table.num_schemas()});
    std::vector<i32> sarity(table.schema_arities().begin(), table.schema_arities().end());
    b.add("schema_arity", DType::I32, shape1(sarity.size()), sarity);
    b.scalar("num_instances", I);
    b.scalar("num_domains", 1);
    b.scalar("num_predicates", P);
    b.scalar("num_domain_predicates", table.num_domain_predicates());
    b.scalar("num_schemas", table.num_schemas());
    b.scalar("max_arity", max_arity);
    b.scalar("label_width", table.label_width());
    b.scalar("words", table.words());
    return b.build();
}
}  // namespace mymyr::rl
