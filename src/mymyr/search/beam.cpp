// Beam search (best_first.hpp): layer-synchronous, keeping the best beam_width new successors of each layer by
// (h, g, generation order).

#include "best_first_detail.hpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace mymyr::search
{
namespace
{
using namespace bf;
constexpr f64 k_inf = std::numeric_limits<f64>::infinity();

template<class S, class Q>
struct Beam
{
    struct Candidate
    {
        f64 h, g;
        u64 order;
        u32 id;
        u64 woff;
        u32 nw;
    };

    void operator()(Context& c) const
    {
        BestFirstResult& r = c.r;
        r.store = S::k_name;
        r.queue = "layer";
        S store(c.task);
        Nodes nodes;
        nodes.track_depth = c.o.control.budget.max_depth != std::numeric_limits<u32>::max();
        Transitions tr;
        Cur cur;
        std::vector<u64> next, cand_words;
        std::vector<Candidate> cand;
        std::vector<u32> layer, next_layer, pend;  // pend: candidates awaiting a batched evaluation
        std::vector<StateView> batch;
        std::vector<f64> batch_h;
        auto word_view = [&](const Candidate& x) -> StateView
        {
            const u64* w = cand_words.data() + x.woff;
            return {w, x.nw, c.nn ? w + x.nw : nullptr, c.nn};
        };
        auto finish = [&]
        {
            c.finish(store.size(), store.bytes() + nodes.bytes() + cand.capacity() * sizeof(Candidate) +
                                       cand_words.capacity() * 8 + (layer.capacity() + next_layer.capacity()) * 4);
        };

        const StateView s0 = c.start.view();
        const u32 root = store.insert(s0).first;
        nodes.add(k_none, k_none, nullptr, 0, c.g0);
        if (c.unsolvable())
        {
            c.begin_search();
            r.status = SearchStatus::Unsolvable;
            return finish();
        }
        if (c.is_goal(s0))
        {
            c.begin_search();
            c.solved(nodes, root, s0, true);
            return finish();
        }
        const f64 h0 = c.evaluate(s0);
        r.initial_h = h0;
        c.begin_search();
        if (h0 == k_inf)
        {
            ++r.dead_ends;
            r.status = SearchStatus::Unsolvable;
            return finish();
        }
        nodes.h[root] = h0;
        nodes.set(root, Nodes::Evaluated);
        nodes.set_status(root, Nodes::Open);
        layer.push_back(store.make_ref(root, s0));
        const u32 width = std::max<u32>(1, c.o.beam_width);
        const u32 max_depth = c.o.control.budget.max_depth;
        u64 order = 0;
        bool stop = false;
        while (!layer.empty() && !stop)
        {
            ++r.layers;
            cand.clear();
            cand_words.clear();
            usize li = 0;
            for (; li < layer.size() && !stop; ++li)
            {
                if (!c.keep_going())
                {
                    stop = true;
                    break;
                }
                const u32 ref = layer[li];
                const u32 id = store.load(ref, cur);
                store.release(ref);
                nodes.set_status(id, Nodes::Closed);
                if (nodes.track_depth && nodes.depth[id] >= max_depth)
                {
                    ++r.stats.pruned;
                    continue;
                }
                ++r.stats.expanded;
                if (c.obs)
                    c.obs->on_expand(id, cur.view());
                if (!expand(c, store, nodes, id, cur, next, tr, false))
                {
                    stop = true;  // r.status is set
                    break;
                }
                for (const Transitions::T& t : tr.t)
                {
                    const u32 cid = t.child;
                    if (c.obs)
                        c.obs->on_generate(id, c.action(t.schema, tr.binding(t)), cid, tr.view(t), t.fresh);
                    const StateView v = tr.view(t);
                    bool pending = false;
                    if (t.fresh)
                    {
                        if (c.is_goal(v))
                        {
                            nodes.set(cid, Nodes::Goal);
                            c.solved(nodes, cid, v, true);
                            stop = true;
                            break;
                        }
                        if (c.batched)
                            pending = true;  // evaluated with the rest of the layer
                        else
                        {
                            const f64 hv = c.evaluate(v);
                            if (hv == k_inf)
                            {
                                nodes.set_status(cid, Nodes::DeadEnd);
                                ++r.dead_ends;
                                ++r.stats.pruned;
                                continue;
                            }
                            nodes.h[cid] = hv;
                            nodes.set(cid, Nodes::Evaluated);
                        }
                    }
                    else
                    {
                        // a state an earlier layer generated but did not keep may compete again
                        if (nodes.status(cid) != Nodes::New || nodes.has(cid, Nodes::Candidate))
                            continue;
                        const f64 gc = t.g;
                        if (gc < nodes.g[cid])
                            nodes.relink(cid, id, t.schema, tr.binding(t), c.succ.arity(t.schema), gc);
                    }
                    nodes.set(cid, Nodes::Candidate);
                    Candidate x{nodes.h[cid], nodes.g[cid], order++, cid, cand_words.size(), 0};
                    if (S::k_needs_words || pending)
                    {
                        x.nw = v.nw;
                        cand_words.insert(cand_words.end(), v.w, v.w + v.nw);
                        if (c.nn)
                            cand_words.insert(cand_words.end(), v.num, v.num + c.nn);
                    }
                    if (pending)
                        pend.push_back(static_cast<u32>(cand.size()));
                    cand.push_back(x);
                }
            }
            for (; li < layer.size(); ++li)
                store.release(layer[li]);
            if (stop)
                break;
            if (!pend.empty())
            {
                // batched evaluation of the layer's new successors; dead ends leave the candidates
                batch.clear();
                for (u32 i : pend)
                    batch.push_back(word_view(cand[i]));
                batch_h.resize(batch.size());
                c.evaluate(batch, batch_h);
                for (usize j = 0; j < pend.size(); ++j)
                {
                    Candidate& x = cand[pend[j]];
                    x.h = batch_h[j];
                    if (x.h == k_inf)
                    {
                        nodes.unset(x.id, Nodes::Candidate);
                        nodes.set_status(x.id, Nodes::DeadEnd);
                        ++r.dead_ends;
                        ++r.stats.pruned;
                        continue;
                    }
                    nodes.h[x.id] = x.h;
                    nodes.set(x.id, Nodes::Evaluated);
                }
                pend.clear();
                std::erase_if(cand, [](const Candidate& x) { return x.h == k_inf; });
            }
            // the best `width` candidates by (h, g, generation order): a total order
            auto better = [](const Candidate& a, const Candidate& b)
            {
                if (a.h != b.h)
                    return a.h < b.h;
                if (a.g != b.g)
                    return a.g < b.g;
                return a.order < b.order;
            };
            if (cand.size() > width)
                std::nth_element(cand.begin(), cand.begin() + width, cand.end(), better);
            const usize keep = std::min<usize>(width, cand.size());
            std::sort(cand.begin(), cand.begin() + static_cast<std::ptrdiff_t>(keep), better);
            next_layer.clear();
            for (usize i = 0; i < cand.size(); ++i)
            {
                const Candidate& x = cand[i];
                nodes.unset(x.id, Nodes::Candidate);
                if (i < keep)
                {
                    nodes.set_status(x.id, Nodes::Open);
                    next_layer.push_back(store.make_ref(x.id, word_view(x)));
                }
            }
            layer.swap(next_layer);
        }
        if (!stop)
            r.status = SearchStatus::Exhausted;  // the beam ran dry
        finish();
    }
};
}  // namespace

BestFirstResult beam(const Task& task, const BestFirstOptions& options)
{
    return bf::run<Beam>(task, options, "beam");
}
}  // namespace mymyr::search
