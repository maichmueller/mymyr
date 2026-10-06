#include "golden.hpp"

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

/// The domain file next to the problem.
std::optional<GoldenTask> inst(const fs::path& root, const std::string& tag, const std::string& problem)
{
    const fs::path dir = root / tag;
    const std::string stem = fs::path(problem).stem().string();
    GoldenTask t;
    t.name = fs::path(tag).filename().string() + "__" + stem;
    for (const auto& cand : {"domain_" + stem + ".pddl", "domain-" + stem + ".pddl", std::string("domain.pddl")})
        if (fs::exists(dir / cand))
        {
            t.domain = dir / cand;
            break;
        }
    t.problem = dir / problem;
    if (t.domain.empty() || !fs::exists(t.problem))
        return std::nullopt;
    return t;
}
}  // namespace

fs::path test_data_dir() { return fs::path(MYMYR_TEST_DATA_DIR); }

fs::path fork_data_dir()
{
    if (const char* d = std::getenv("MYMYR_FORK_DATA"); d && *d)
        return d;
    return fs::path(MYMYR_FORK_DATA_DIR);
}

std::vector<GoldenTask> ipc_tasks(std::vector<std::string>* missing)
{
    std::vector<GoldenTask> out;
    const fs::path dir = test_data_dir() / "golden/ipc";
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

fs::path work_dir()
{
    if (const char* w = std::getenv("MYMYR_WORK"); w && *w)
        return w;
    return fs::path(MYMYR_WORK_DIR);
}

std::vector<GoldenTask> suite_tasks(std::vector<std::string>* missing)
{
    const fs::path bench = work_dir() / "mimir-cs" / "Benchmark";
    const fs::path patched = test_data_dir() / "pddl";
    struct Entry
    {
        const char* tag;
        const char* problem;
        bool patched;
    };
    // The 22-task BrFS suite (export_all.py's BFS_INSTANCES)
    const Entry entries[] = {
        {"strips/gripper", "prob05.pddl", false},
        {"strips/blocks", "probBLOCKS-8-0.pddl", false},
        {"strips/logistics00", "probLOGISTICS-6-1.pddl", false},
        {"strips/miconic", "s7-4.pddl", false},
        {"strips/visitall", "visitall_x-6_y-3_r-100.pddl", false},
        {"strips/sokoban-opt08-strips", "p14.pddl", false},
        {"strips/depot", "p02.pddl", false},
        {"strips/driverlog", "p03.pddl", false},
        {"strips/rovers", "p02.pddl", false},
        {"strips/zenotravel", "p05.pddl", false},
        {"strips/transport-opt08-strips", "p23.pddl", false},
        {"strips/freecell", "p02.pddl", false},
        {"strips/snake-opt18-strips", "p05.pddl", false},
        {"strips/parcprinter-opt11-strips", "p03.pddl", false},
        {"strips/pegsol-08-strips", "p22.pddl", false},
        {"adl/miconic-simpleadl", "s10-2.pddl", false},
        {"adl/caldera-split-opt18-adl", "p04.pddl", false},
        {"adl/pathways", "p02.pddl", false},
        {"adl/folding-opt23-adl", "p01.pddl", false},
        {"adl/openstacks-opt08-adl", "p03.pddl", false},
        {"philosophers", "p03-phil4.pddl", true},
        {"strips/organic-synthesis-opt18-strips", "p20.pddl", false},
    };
    std::vector<GoldenTask> out;
    for (const auto& e : entries)
    {
        auto t = inst(e.patched ? patched : bench, e.tag, e.problem);
        if (!t)
        {
            if (missing)
                missing->push_back(std::string(e.tag) + "/" + e.problem);
            continue;
        }
        add_goldens(*t, {test_data_dir() / "golden/suite", test_data_dir() / "golden/suite-numeric",
                         test_data_dir() / "numeric_tasks/suite", test_data_dir() / "tasks"});
        out.push_back(std::move(*t));
    }
    return out;
}

std::vector<GoldenTask> numeric_tasks(std::vector<std::string>* missing)
{
    // The numeric task suite
    const fs::path csn = work_dir() / "mimir-cs" / "Benchmark" / "numeric";
    const fs::path md = work_dir() / "mimir" / "data";
    std::vector<GoldenTask> out;
    auto add = [&](std::string name, fs::path domain, fs::path problem)
    {
        if (!fs::exists(domain) || !fs::exists(problem))
        {
            if (missing)
                missing->push_back(name);
            return;
        }
        GoldenTask t{std::move(name), std::move(domain), std::move(problem), {}};
        add_goldens(t, {test_data_dir() / "golden/numeric", test_data_dir() / "numeric_tasks"});
        out.push_back(std::move(t));
    };
    for (const char* d : {"block-grouping", "counters", "delivery", "drone", "expedition", "ext-plant-watering", "farmland",
                          "hydropower", "sailing", "tpp"})
        add(std::string("cs-") + d, csn / d / "domain.pddl", csn / d / "pfile1.pddl");
    for (const char* d : {"fo-counters", "refuel", "refuel-adl", "tpp/numeric", "zenotravel/numeric", "woodworking", "barman", "transport"})
    {
        std::string name = std::string("m-") + d;
        std::replace(name.begin(), name.end(), '/', '-');
        add(name, md / d / "domain.pddl", md / d / "test_problem.pddl");
    }
    const fs::path pddl = test_data_dir() / "numeric_tasks/pddl";
    add("cs-block-grouping-conj", csn / "block-grouping" / "domain.pddl", pddl / "block-grouping-conj.pddl");
    add("cs-delivery-nometric", csn / "delivery" / "domain.pddl", pddl / "delivery-nometric.pddl");
    return out;
}

std::vector<std::string> tokens(const std::string& text)
{
    std::istringstream in(text);
    std::vector<std::string> out;
    std::string w;
    while (in >> w)
        out.push_back(w);
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
