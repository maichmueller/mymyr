// TaskSuite: instances of several domains (include/mymyr/rl/task_suite.hpp).

#include "mymyr/rl/task_suite.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/core/hash.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace mymyr::rl
{
TaskSuitePtr TaskSuite::create(std::vector<TaskTablePtr> tables)
{
    std::vector<Ref> refs;
    if (tables.size() > 1)
        for (u32 d = 0; d < tables.size(); ++d)
            if (tables[d])
                for (u32 i = 0; i < tables[d]->size(); ++i)
                    refs.push_back({d, i});
    return std::make_shared<const TaskSuite>(Private{}, std::move(tables), std::move(refs));
}

TaskSuitePtr TaskSuite::group(std::vector<TaskPtr> tasks)
{
    if (tasks.empty())
        throw std::invalid_argument("mymyr: TaskSuite: no instances");
    for (usize i = 0; i < tasks.size(); ++i)
        if (!tasks[i])
            throw std::invalid_argument("mymyr: TaskSuite: instance " + std::to_string(i) + " is null");
    // each task joins the first domain it belongs to (its first task is the domain's reference)
    std::vector<std::vector<TaskPtr>> groups;
    std::vector<Ref> refs;
    refs.reserve(tasks.size());
    for (TaskPtr& t : tasks)
    {
        u32 d = 0;
        while (d < groups.size() && !TaskTable::domain_mismatch(*groups[d].front(), *t).empty())
            ++d;
        if (d == groups.size())
            groups.emplace_back();
        refs.push_back({d, static_cast<u32>(groups[d].size())});
        groups[d].push_back(std::move(t));
    }
    std::vector<TaskTablePtr> tables;
    tables.reserve(groups.size());
    for (auto& g : groups)
        tables.push_back(TaskTable::create(std::move(g)));
    if (tables.size() == 1)
        refs.clear();  // one domain: global ids are the table's
    return std::make_shared<const TaskSuite>(Private{}, std::move(tables), std::move(refs));
}

TaskSuitePtr TaskSuite::of(TaskTablePtr table)
{
    std::vector<TaskTablePtr> v;
    v.push_back(std::move(table));
    return std::make_shared<const TaskSuite>(Private{}, std::move(v), std::vector<Ref>{});
}

TaskSuite::TaskSuite(Private, std::vector<TaskTablePtr> tables, std::vector<Ref> refs)
    : m_tables(std::move(tables)), m_refs(std::move(refs))
{
    if (m_tables.empty())
        throw std::invalid_argument("mymyr: TaskSuite: no domains");
    u64 total = 0;
    for (usize d = 0; d < m_tables.size(); ++d)
    {
        if (!m_tables[d])
            throw std::invalid_argument("mymyr: TaskSuite: table " + std::to_string(d) + " is null");
        total += m_tables[d]->size();
    }
    if (total > static_cast<u64>(std::numeric_limits<i32>::max()))
        throw std::invalid_argument("mymyr: TaskSuite: more than 2^31 - 1 instances");
    m_size = static_cast<u32>(total);
    for (usize d = 1; d < m_tables.size(); ++d)
        for (usize e = 0; e < d; ++e)
            if (TaskTable::domain_mismatch(*m_tables[e]->task(0), *m_tables[d]->task(0)).empty())
                throw std::invalid_argument("mymyr: TaskSuite: tables " + std::to_string(e) + " and " + std::to_string(d) +
                                            " are of one domain (" + m_tables[d]->task(0)->data().domain_name +
                                            "): build one table of their instances");
    if (!m_refs.empty())
    {
        if (m_refs.size() != m_size)
            throw std::invalid_argument("mymyr: TaskSuite: an id map of " + std::to_string(m_refs.size()) +
                                        " instances, the tables have " + std::to_string(m_size));
        m_global.resize(m_tables.size());
        for (usize d = 0; d < m_tables.size(); ++d)
            m_global[d].assign(m_tables[d]->size(), ~u32{0});
        for (u32 g = 0; g < m_size; ++g)
        {
            const Ref r = m_refs[g];
            if (r.domain >= m_tables.size() || r.local >= m_tables[r.domain]->size() ||
                m_global[r.domain][r.local] != ~u32{0})
                throw std::invalid_argument("mymyr: TaskSuite: the id map is not a bijection");
            m_global[r.domain][r.local] = g;
        }
    }
    m_schema_offsets.push_back(0);
    for (const TaskTablePtr& t : m_tables)
    {
        m_words = std::max(m_words, t->words());
        m_numeric_words = std::max(m_numeric_words, t->numeric_words());
        m_label_width = std::max(m_label_width, t->label_width());
        m_max_objects = std::max(m_max_objects, t->max_objects());
        m_max_schemas = std::max(m_max_schemas, t->num_schemas());
        m_numeric = m_numeric || t->numeric();
        m_schema_offsets.push_back(m_schema_offsets.back() + t->num_schemas());
    }
}

std::string TaskSuite::domain_name(u32 domain) const { return table(domain)->task(0)->data().domain_name; }

void TaskSuite::initial_row(u32 g, u64* row, u32 words, u32 numeric_words) const
{
    const Ref r = ref(g);
    const TaskTable& t = *m_tables[r.domain];
    const std::vector<u64>& init = t.instance(r.local).init;  // [t.words() | t.numeric_words()]
    std::fill_n(row, u64{words} + numeric_words, u64{0});
    std::copy_n(init.begin(), t.words(), row);
    std::copy_n(init.begin() + t.words(), t.numeric_words(), row + words);
}

u64 TaskSuite::fingerprint() const
{
    if (single_domain())
        return m_tables[0]->fingerprint();
    u64 h = hash::combine(hash::k_seed ^ m_tables.size(), m_size);
    for (const TaskTablePtr& t : m_tables)
        h = hash::combine(h, t->fingerprint());
    for (const Ref r : m_refs)
        h = hash::combine(h, (u64{r.domain} << 32) | r.local);
    return h;
}

void TaskSuite::check_task_ids(const i32* task_ids, u64 n) const
{
    if (single_domain())
    {
        m_tables[0]->check_task_ids(task_ids, n);
        return;
    }
    if (!task_ids)
        return;
    const i64 I = m_size;
    for (u64 r = 0; r < n; ++r)
        if (task_ids[r] < 0 || task_ids[r] >= I)
            throw std::invalid_argument("mymyr: task id " + std::to_string(task_ids[r]) + " of row " + std::to_string(r) +
                                        " is outside the suite's " + std::to_string(I) + " instances");
}

ArrayBundle suite_atom_metadata(const TaskSuite& suite)
{
    const u32 I = suite.size(), D = suite.num_domains();
    std::vector<ArrayBundle> per;  // each domain's table metadata
    per.reserve(D);
    for (u32 d = 0; d < D; ++d)
        per.push_back(table_atom_metadata(*suite.table(d)));
    if (D == 1)
        return std::move(per[0]);
    auto arr = [](const ArrayBundle& b, const std::string& name) -> const ArrayInfo&
    {
        const ArrayInfo* a = b.find(name);
        if (!a)
            throw std::logic_error("mymyr: suite_atom_metadata: no " + name + " (internal error)");
        return *a;
    };
    auto i32s = [&](const ArrayBundle& b, const std::string& name) { return static_cast<const i32*>(b.data(arr(b, name))); };
    auto i64s = [&](const ArrayBundle& b, const std::string& name) { return static_cast<const i64*>(b.data(arr(b, name))); };
    ArrayBundle::Builder out;
    auto shape1 = [](u64 n) { return std::vector<i64>{static_cast<i64>(n)}; };
    auto shape2 = [](u64 n, u64 m) { return std::vector<i64>{static_cast<i64>(n), static_cast<i64>(m)}; };
    // the atoms of every instance in global order (each domain's rows of its instance), args padded to the widest
    for (const char* prefix : {"atom", "derived", "static"})
    {
        const std::string p(prefix);
        const bool cid = p != "static";
        u32 A = 0;
        for (u32 d = 0; d < D; ++d)
        {
            const ArrayInfo& a = arr(per[d], p + "_args");
            A = std::max<u32>(A, a.shape.size() == 2 ? static_cast<u32>(a.shape[1]) : 0);
        }
        std::vector<i64> offsets{0};
        std::vector<i32> pred, args, cids;
        for (u32 g = 0; g < I; ++g)
        {
            const TaskSuite::Ref r = suite.ref(g);
            const ArrayBundle& m = per[r.domain];
            const i64* off = i64s(m, p + "_offsets");
            const ArrayInfo& aa = arr(m, p + "_args");
            const u32 Ad = aa.shape.size() == 2 ? static_cast<u32>(aa.shape[1]) : 0;
            const i32* mp = i32s(m, p + "_pred");
            const auto* ma = static_cast<const i32*>(m.data(aa));
            for (i64 j = off[r.local]; j < off[r.local + 1]; ++j)
            {
                pred.push_back(mp[j]);
                for (u32 k = 0; k < A; ++k)
                    args.push_back(k < Ad ? ma[static_cast<u64>(j) * Ad + k] : -1);
                if (cid)
                    cids.push_back(i32s(m, p + "_cid")[j]);
            }
            offsets.push_back(offsets.back() + (off[r.local + 1] - off[r.local]));
        }
        out.add(p + "_offsets", DType::I64, shape1(offsets.size()), offsets);
        out.add(p + "_pred", DType::I32, shape1(pred.size()), pred);
        out.add(p + "_args", DType::I32, shape2(pred.size(), A), args);
        if (cid)
            out.add(p + "_cid", DType::I32, shape1(cids.size()), cids);
    }
    std::vector<i32> num_objects(I), num_atoms(I), words(I), domain(I), local(I);
    std::vector<i64> object_offsets(I + 1, 0);
    for (u32 g = 0; g < I; ++g)
    {
        const TaskSuite::Ref r = suite.ref(g);
        num_objects[g] = i32s(per[r.domain], "num_objects")[r.local];
        num_atoms[g] = i32s(per[r.domain], "num_atoms")[r.local];
        words[g] = i32s(per[r.domain], "words")[r.local];
        domain[g] = static_cast<i32>(r.domain);
        local[g] = static_cast<i32>(r.local);
        object_offsets[g + 1] = object_offsets[g] + num_objects[g];
    }
    out.add("num_objects", DType::I32, shape1(I), num_objects);
    out.add("object_offsets", DType::I64, shape1(I + 1), object_offsets);
    out.add("num_atoms", DType::I32, shape1(I), num_atoms);
    out.add("words", DType::I32, shape1(I), words);
    out.add("domain", DType::I32, shape1(I), domain);
    out.add("local_id", DType::I32, shape1(I), local);
    // each domain's predicates (problem-local ones name the global instance), and its schemas
    std::vector<i32> parity, pkind, pinst, dpreds(D), sarity;
    std::vector<i64> poff{0}, soff{0};
    u32 max_arity = 0;
    for (u32 d = 0; d < D; ++d)
    {
        const TaskTable& t = *suite.table(d);
        for (usize p = 0; p < t.predicate_names().size(); ++p)
        {
            parity.push_back(static_cast<i32>(t.predicate_arities()[p]));
            pkind.push_back(t.predicate_kinds()[p]);
            const i32 owner = t.predicate_instances()[p];
            pinst.push_back(owner < 0 ? -1 : static_cast<i32>(suite.global_id(d, static_cast<u32>(owner))));
            max_arity = std::max(max_arity, t.predicate_arities()[p]);
        }
        poff.push_back(static_cast<i64>(parity.size()));
        dpreds[d] = static_cast<i32>(t.num_domain_predicates());
        for (const u32 a : t.schema_arities())
            sarity.push_back(static_cast<i32>(a));
        soff.push_back(static_cast<i64>(sarity.size()));
    }
    const auto P = static_cast<u32>(parity.size());
    out.add("pred_offsets", DType::I64, shape1(D + 1), poff);
    out.add("pred_arity", DType::I32, shape1(P), parity);
    out.add("pred_kind", DType::I32, shape1(P), pkind);
    out.add("pred_instance", DType::I32, shape1(P), pinst);
    out.add("domain_predicates", DType::I32, shape1(D), dpreds);
    out.add("schema_offsets", DType::I64, shape1(D + 1), soff);
    out.add("schema_arity", DType::I32, shape1(sarity.size()), sarity);
    out.scalar("num_instances", I);
    out.scalar("num_domains", D);
    out.scalar("num_predicates", P);
    out.scalar("num_domain_predicates", suite.table(0)->num_domain_predicates());
    out.scalar("num_schemas", suite.table(0)->num_schemas());
    out.scalar("max_arity", max_arity);
    out.scalar("label_width", suite.label_width());
    out.scalar("words", suite.words());
    return out.build();
}
}  // namespace mymyr::rl
