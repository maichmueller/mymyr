#include "golden.hpp"

#include "../../cpp/support/benchmarks.hpp"
#include "mymyr/formalism/text_format.hpp"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <sstream>

namespace mymyr::test
{
namespace fs = std::filesystem;
using namespace formalism;

namespace
{
void add_goldens(GoldenTask& t, std::initializer_list<fs::path> dirs)
{
    for (const auto& d : dirs)
        if (const fs::path p = d / (t.name + ".txt"); fs::exists(p))
            t.goldens.push_back(p);
}

}  // namespace

fs::path parity_data_dir()
{
    if (const char* d = std::getenv("MYMYR_PARITY_DATA"); d && *d)
        return d;
    return fs::path(MYMYR_PARITY_DATA_DIR);
}

fs::path fork_data_dir() { return parity_data_dir() / "mimir-fork" / "data"; }

fs::path work_dir() { return parity_data_dir(); }

fs::path golden_dir() { return fs::path(MYMYR_PARITY_TEST_DATA_DIR) / "golden"; }

std::vector<GoldenTask> ipc_tasks(std::vector<std::string>* missing)
{
    std::vector<GoldenTask> out;
    const fs::path dir = golden_dir() / "ipc";
    if (!fs::exists(dir))
        return out;
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".txt")
            files.push_back(e.path());
    std::sort(files.begin(), files.end());
    for (const auto& f : files)
    {
        // <domain>__<split>__<problem>.txt
        const std::string name = f.stem().string();
        const size_t a = name.find("__"), b = name.rfind("__");
        if (a == std::string::npos || a == b)
            continue;
        const std::string domain = name.substr(0, a), split = name.substr(a + 2, b - a - 2), problem = name.substr(b + 2);
        fs::path base;
        for (const auto& cand : {fork_data_dir() / "ipc" / (domain + "-ipc"), fork_data_dir() / "ipc" / domain})
            if (fs::exists(cand / split / (problem + ".pddl")))
                base = cand / split;
        if (base.empty())
        {
            if (missing)
                missing->push_back(name);
            continue;
        }
        out.push_back(GoldenTask{name, base / "domain.pddl", base / (problem + ".pddl"), {f}});
    }
    return out;
}

std::vector<GoldenTask> suite_tasks(std::vector<std::string>* missing)
{
    std::vector<GoldenTask> out;
    for (auto& t : test::classic_suite(missing))
    {
        GoldenTask g{t.name, t.domain, t.problem, {}};
        add_goldens(g, {golden_dir() / "suite", golden_dir() / "suite-numeric", test::test_data_dir() / "numeric_tasks/suite",
                        test::test_data_dir() / "tasks"});
        out.push_back(std::move(g));
    }
    return out;
}

std::vector<GoldenTask> numeric_tasks(std::vector<std::string>* missing)
{
    std::vector<GoldenTask> out;
    for (auto& t : test::numeric_suite(missing))
    {
        GoldenTask g{t.name, t.domain, t.problem, {}};
        add_goldens(g, {golden_dir() / "numeric", test::test_data_dir() / "numeric_tasks"});
        out.push_back(std::move(g));
    }
    return out;
}

std::set<std::string> type_predicates(const TaskData& t)
{
    std::set<std::string> types;
    for (const auto& ty : t.types)
        types.emplace(t.str(ty.name));
    std::set<std::string> out;
    for (const auto& p : t.predicates)
        if (p.kind == PredKind::Static && p.arity == 1 && types.contains(std::string(t.str(p.name))))
            out.emplace(t.str(p.name));
    return out;
}

namespace
{
}  // namespace

bool is_generated_axiom_predicate(const TaskData& t, u32 p)
{
    const Predicate& pr = t.predicates[p];
    const std::string_view n = t.str(pr.name);
    return pr.kind == PredKind::Derived && pr.arity >= 2 && n.size() > 6 && n.substr(0, 6) == "axiom_"
           && n.find_first_not_of("0123456789", 6) == std::string_view::npos;
}

namespace
{
std::string expr_text(const TaskData& t, u32 e)
{
    const Expr& x = t.exprs[e];
    switch (x.op)
    {
        case ExprOp::Number: return "c" + std::to_string(x.value);
        case ExprOp::Function:
        {
            std::string s = "f" + std::to_string(x.func.v);
            for (Term tt : TaskData::slice(t.terms, x.terms))
                s += "," + std::to_string(tt);
            return s;
        }
        case ExprOp::Neg: return "(-" + expr_text(t, x.a) + ")";
        default: return "(" + expr_text(t, x.a) + std::to_string(static_cast<int>(x.op)) + expr_text(t, x.b) + ")";
    }
}

}  // namespace

std::string canonical_text(const TaskData& in, const std::set<std::string>& type_preds)
{
    TaskData t = in;
    const u32 np = static_cast<u32>(in.predicates.size());
    auto name_of = [&](u32 p) { return std::string(t.str(t.predicates[p].name)); };
    auto is_type = [&](u32 p) { return t.predicates[p].arity == 1 && type_preds.contains(name_of(p)); };

    // sorts every maximal run [i, j) of consecutive items with the same non-empty run key by `name`
    auto sort_runs = [](auto first, auto last, auto run_key, auto name)
    {
        for (auto i = first; i != last;)
        {
            const auto k = run_key(*i);
            auto j = std::next(i);
            if (k)
                while (j != last && run_key(*j) == k)
                    ++j;
            if (std::distance(i, j) > 1)
                std::stable_sort(i, j, [&](const auto& a, const auto& b) { return name(a) < name(b); });
            i = j;
        }
    };

    // 1. predicates: runs of type predicates by name; remap every reference
    std::vector<u32> order(np);
    std::iota(order.begin(), order.end(), 0u);
    sort_runs(order.begin(), order.end(), [&](u32 p) { return is_type(p) ? std::optional<int>(0) : std::nullopt; },
              [&](u32 p) { return name_of(p); });
    std::vector<u32> new_id(np);
    std::vector<Predicate> preds(np);
    for (u32 i = 0; i < np; ++i)
    {
        new_id[order[i]] = i;
        preds[i] = in.predicates[order[i]];
    }
    t.predicates = std::move(preds);
    for (auto& l : t.literals)
        l.pred = PredicateId{new_id[l.pred.v]};
    for (auto& a : t.static_init)
        a.pred = PredicateId{new_id[a.pred.v]};
    for (auto& a : t.fluent_init)
        a.pred = PredicateId{new_id[a.pred.v]};
    for (auto& x : t.axioms)
        x.head.pred = PredicateId{new_id[x.head.pred.v]};

    // 2. literal lists: runs of type literals over one term
    auto lit_key = [&](const Literal& l) -> std::optional<Term>
    {
        if (!l.positive || l.terms.count != 1 || !is_type(l.pred.v))
            return std::nullopt;
        return t.terms[l.terms.begin];
    };
    auto lit_name = [&](const Literal& l) { return name_of(l.pred.v); };
    auto sort_range = [&](Range r)
    {
        auto b = t.literals.begin() + r.begin;
        sort_runs(b, b + r.count, lit_key, lit_name);
    };
    for (const auto& s : t.schemas)
    {
        sort_range(s.precondition.literals);
        for (const auto& ce : TaskData::slice(t.conditional_effects, s.effects))
        {
            sort_range(ce.condition.literals);
            sort_range(ce.effects);
        }
    }
    for (const auto& x : t.axioms)
    {
        if (is_generated_axiom_predicate(t, x.head.pred.v))
        {
            // loki types the parameters of a generated axiom in their address-dependent order: runs of type literals
            // over any parameters, by (parameter, name)
            auto b = t.literals.begin() + x.body.literals.begin;
            sort_runs(b, b + x.body.literals.count,
                      [&](const Literal& l) { return lit_key(l) ? std::optional<int>(0) : std::nullopt; },
                      [&](const Literal& l) { return std::make_pair(t.terms[l.terms.begin], name_of(l.pred.v)); });
        }
        else
            sort_range(x.body.literals);
    }
    sort_range(t.goal.literals);

    // 2b. numeric effects of one conditional effect: a multiset (loki creates the summed effects in the iteration
    // order of an unordered_map keyed by Function pointers)
    for (const auto& ce : t.conditional_effects)
    {
        auto b = t.numeric_effects.begin() + ce.numeric_effects.begin;
        std::stable_sort(b, b + ce.numeric_effects.count, [&](const NumericEffect& x, const NumericEffect& y)
                         {
                             auto key = [&](const NumericEffect& e)
                             {
                                 std::string k = std::to_string(e.func.v) + "/" + std::to_string(static_cast<int>(e.op));
                                 for (Term tt : TaskData::slice(t.terms, e.terms))
                                     k += "," + std::to_string(tt);
                                 return k + "/" + expr_text(t, e.expr);
                             };
                             return key(x) < key(y);
                         });
    }

    // 3. initial atoms: runs of type atoms of one object
    auto atom_key = [&](const GroundAtom& a) -> std::optional<u32>
    {
        if (a.objects.count != 1 || !is_type(a.pred.v))
            return std::nullopt;
        return t.object_ids[a.objects.begin].v;
    };
    auto atom_name = [&](const GroundAtom& a) { return name_of(a.pred.v); };
    sort_runs(t.static_init.begin(), t.static_init.end(), atom_key, atom_name);
    sort_runs(t.fluent_init.begin(), t.fluent_init.end(), atom_key, atom_name);

    return write_task_text(t);
}

std::optional<std::string> first_difference(const std::vector<std::string>& a, const std::vector<std::string>& b)
{
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i)
        if (a[i] != b[i])
        {
            std::string ctx;
            for (size_t k = i >= 6 ? i - 6 : 0; k < std::min(n, i + 6); ++k)
                ctx += (k == i ? "[" + a[k] + "|" + b[k] + "]" : a[k]) + " ";
            return "token " + std::to_string(i) + ": " + ctx;
        }
    if (a.size() != b.size())
        return "lengths " + std::to_string(a.size()) + " vs " + std::to_string(b.size());
    return std::nullopt;
}
}  // namespace mymyr::test
