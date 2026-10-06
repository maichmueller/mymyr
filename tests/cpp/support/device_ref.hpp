#pragma once
// Reference data for the device task layout (rl::device_arrays version 2, rl/task_arrays_view.hpp): random-walk
// states, action labels to test (every applicable action plus perturbed and random labels), and the CPU engine's
// answers (is_applicable, apply, is_goal, derived bitsets). Shared by the host test of the layout and the CUDA smoke
// test (tests/cuda), which must reproduce these answers on the device.

#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <random>
#include <vector>

namespace mymyr::test
{
struct DeviceRef
{
    u32 W = 0;   // state words of the batch (the widest state, at least the export's state_words)
    u32 DW = 0;  // derived words (0 when the task has no axioms)
    u32 L = 0;   // label columns (the largest schema arity)
    std::vector<u64> states;    // [N, W]
    std::vector<u64> derived;   // [N, DW], from the CPU axiom evaluator
    std::vector<u8> goal;       // [N]
    // labels: pairs (state index, schema, binding)
    std::vector<u32> label_state, label_schema;
    std::vector<u32> label_binding;  // [K, L], unused columns zero
    std::vector<u8> applicable;      // [K], CPU is_applicable
    std::vector<u64> successor;      // [K, W], CPU apply (zero rows where not applicable)
    [[nodiscard]] u64 n() const { return goal.size(); }
    [[nodiscard]] u64 k() const { return label_schema.size(); }
};

/// Seeded random walks (canonical order, witness pruning off), the initial state first.
inline std::vector<State> device_ref_walks(const Task& task, u32 walks, u32 steps, u64 seed)
{
    std::mt19937_64 rng(seed);
    Successors& succ = task.workspace().successors();
    std::vector<State> out;
    std::vector<u64> tmp;
    for (u32 w = 0; w < walks; ++w)
    {
        State s = task.initial_state();
        out.push_back(s);
        for (u32 k = 0; k < steps; ++k)
        {
            std::vector<State> kids;
            succ.prepare(s);
            succ.generate<false>(
                [&](u32, const ObjectId*, const Delta& d)
                {
                    const u32 n = apply_delta(s.data(), s.size_words(), d, tmp);
                    kids.emplace_back(tmp.data(), n);
                    return true;
                },
                false, true);
            if (kids.empty())
                break;
            s = kids[rng() % kids.size()];
            out.push_back(s);
        }
    }
    return out;
}

/// Builds the reference for `states`: per state every applicable action (at most `max_per_state`, spread over the
/// canonical order) plus as many perturbed labels (one binding position replaced by a random object) and a few random
/// labels. Take the export afterwards: under lazy slots, apply interns the successors' atoms, so they have slots in
/// the export's snapshot.
inline DeviceRef device_ref(const Task& task, const std::vector<State>& states, u32 min_words, u64 seed,
                            u32 max_per_state = 64)
{
    DeviceRef r;
    std::mt19937_64 rng(seed);
    Successors& succ = task.workspace().successors();
    const u32 S = task.num_schemas();
    const u32 n = task.num_objects();
    r.L = 1;
    for (u32 s = 0; s < S; ++s)
        r.L = std::max(r.L, succ.arity(s));
    r.W = std::max<u32>(1, min_words);
    for (const State& s : states)
        r.W = std::max(r.W, s.size_words());
    r.DW = task.has_axioms() ? std::max<u32>(1, bits::words_for(task.atoms().max_derived_slots())) : 0;
    r.derived.assign(states.size() * r.DW, 0);
    for (usize i = 0; i < states.size(); ++i)
    {
        r.goal.push_back(task.is_goal(states[i]) ? 1 : 0);
        if (r.DW)
        {
            succ.prepare(states[i]);
            const detail::Engine& e = succ.engine();
            std::copy_n(e.derived(), std::min(e.derived_words(), r.DW), r.derived.begin() + static_cast<std::ptrdiff_t>(i * r.DW));
        }
    }
    auto add_label = [&](u32 si, u32 schema, const std::vector<ObjectId>& b)
    {
        r.label_state.push_back(si);
        r.label_schema.push_back(schema);
        for (u32 j = 0; j < r.L; ++j)
            r.label_binding.push_back(j < b.size() ? b[j].v : 0);
    };
    for (u32 si = 0; si < states.size(); ++si)
    {
        std::vector<Action> acts;
        succ.set_witness_pruning(false);
        acts = succ.applicable_actions(states[si]);
        succ.set_witness_pruning(true);
        const usize stride = std::max<usize>(1, acts.size() / max_per_state);
        for (usize a = 0; a < acts.size(); a += stride)
        {
            add_label(si, acts[a].schema.v, acts[a].binding);
            std::vector<ObjectId> b = acts[a].binding;
            if (!b.empty() && n > 0)
            {
                b[rng() % b.size()] = ObjectId{static_cast<u32>(rng() % n)};
                add_label(si, acts[a].schema.v, b);
            }
        }
        for (u32 x = 0; x < 4 && S > 0 && n > 0; ++x)
        {
            const u32 schema = static_cast<u32>(rng() % S);
            std::vector<ObjectId> b(succ.arity(schema));
            for (ObjectId& o : b)
                o = ObjectId{static_cast<u32>(rng() % n)};
            add_label(si, schema, b);
        }
    }
    r.applicable.resize(r.k());
    std::vector<State> next(r.k());
    for (u64 k = 0; k < r.k(); ++k)
    {
        const State& s = states[r.label_state[k]];
        const u32 schema = r.label_schema[k];
        std::vector<ObjectId> b(succ.arity(schema));
        for (u32 j = 0; j < b.size(); ++j)
            b[j] = ObjectId{r.label_binding[k * r.L + j]};
        const ActionLabel label{SchemaId{schema}, b};
        r.applicable[k] = succ.is_applicable(s, label) ? 1 : 0;
        if (r.applicable[k])
        {
            next[k] = succ.apply(s, label);  // under lazy slots this interns the successor's atoms
            r.W = std::max(r.W, next[k].size_words());
        }
    }
    // pack with the final width
    std::vector<u64> packed(states.size() * r.W, 0);
    for (usize i = 0; i < states.size(); ++i)
        std::copy_n(states[i].data(), states[i].size_words(), packed.begin() + static_cast<std::ptrdiff_t>(i * r.W));
    r.states = std::move(packed);
    r.successor.assign(r.k() * r.W, 0);
    for (u64 k = 0; k < r.k(); ++k)
        if (r.applicable[k])
            std::copy_n(next[k].data(), next[k].size_words(), r.successor.begin() + static_cast<std::ptrdiff_t>(k * r.W));
    return r;
}
}  // namespace mymyr::test
