// LIW(k) (search/liw.hpp): the IW family engine with the landmark novelty pruner.

#include "mymyr/search/liw.hpp"

#include "iw_family_detail.hpp"

#include "mymyr/task/workspace.hpp"

#include <stdexcept>

namespace mymyr::search
{
namespace detail
{
namespace
{
/// Mimir's LandmarkNoveltyPruningStrategyImpl over novelty::LandmarkNoveltyTable.
class LiwPruner
{
public:
    LiwPruner(const Task& task, const novelty::LandmarkCoordinates& coords, u32 k, const novelty::LandmarkTableOptions& options)
        : m_coords(coords), m_table(k, coords.num_ranks(), std::max<u32>(task.atoms().max_fluent_slots(), 1), options),
          m_pmask(coords.mask_words()), m_smask(coords.mask_words()), m_fmask(coords.mask_words()), m_kmask(coords.mask_words())
    {
        m_table.reserve(std::max<u32>(task.atoms().fluent_slots(), 1));
    }

    bool init(const u64* w, u32 n)
    {
        const bool any = m_coords.mask(w, n, m_pmask.data());
        m_coords.collect(m_pmask.data(), any, m_ranks);
        return m_table.mark_state(w, n, m_ranks);
    }
    void begin(const u64* w, u32 n) { m_pany = m_coords.mask(w, n, m_pmask.data()); }
    bool test(const u64* pw, u32 pn, const u64* cw, u32 cn, std::span<const u32> add, const Delta& d)
    {
        const bool sany = m_coords.child_mask(m_pmask.data(), cw, cn, add, d.del, m_smask.data());
        m_coords.split_masks(m_pmask.data(), m_pany, m_smask.data(), sany, m_fmask.data(), m_kmask.data());
        return m_table.mark_successor(pw, pn, cw, cn, add, m_fmask.data(), m_kmask.data());
    }
    [[nodiscard]] u64 bytes() const { return m_table.bytes(); }

private:
    const novelty::LandmarkCoordinates& m_coords;
    novelty::LandmarkNoveltyTable m_table;
    std::vector<u64> m_pmask, m_smask, m_fmask, m_kmask;
    bool m_pany = false;
    std::vector<u32> m_ranks;
};
}  // namespace
}  // namespace detail

IwResult liw(const Task& task, const LiwOptions& o)
{
    if (task.numeric_slots() > 0)
        throw std::invalid_argument("mymyr: LIW does not support tasks with numeric fluents (IW and SIW do)");
    if (o.max_arity > novelty::k_max_arity)
        return detail::failed("LIW arity " + std::to_string(o.max_arity) + " exceeds the maximum " + std::to_string(novelty::k_max_arity));
    Successors& succ = task.workspace().successors();
    const detail::GoalTest goal = detail::GoalTest::from_spec(task, o.control.goal);
    const detail::BlockedSet blocked(o.control.blocked_states);
    detail::Env env(task, succ, goal, blocked, o.control);
    env.witness = o.witness_pruning;
    env.canonical = o.canonical_order;
    env.successor_order = &o.successor_order;
    detail::LayerOrderer layers;
    std::unique_ptr<detail::BeamTeam> team;
    if (std::string e = detail::apply_layers(env, o.layers, layers, o.threads, team); !e.empty())
        return detail::failed(std::move(e));
    if (o.layers.beam() && o.layers.beam_novelty != LayerOrdering::BeamNovelty::AllTested)
        return detail::failed("LIW supports only LayerOrdering::BeamNovelty::AllTested (as in mimir: the landmark novelty table "
                              "has no read-only test)");
    novelty::LandmarkCoordinates coords;
    try
    {
        coords = detail::make_coordinates(task, o.landmarks);
    }
    catch (const std::invalid_argument& e)
    {
        return detail::failed(e.what());
    }
    const State root = o.start ? *o.start : task.initial_state();
    if (env.root)
        env.root->on_start(root);
    IwResult r = detail::novelty_ladder(env, 0, o.max_arity, false,
                                        [&](u32 k, detail::PassOut& po)
                                        {
                                            if (k == 0)
                                            {
                                                detail::NullPruner p;
                                                detail::novelty_pass(env, root, p,
                                                                     detail::PassConfig{.arity = 0,
                                                                                        .root = detail::RootRule::ArityZero,
                                                                                        .root_only = o.width_zero == WidthZero::RootOnly},
                                                                     po);
                                                return;
                                            }
                                            detail::LiwPruner p(task, coords, k, o.landmarks.tables);
                                            detail::novelty_pass(env, root, p, detail::PassConfig{.arity = k}, po);
                                        });
    detail::finish_result(env, root, r);
    return r;
}
}  // namespace mymyr::search
