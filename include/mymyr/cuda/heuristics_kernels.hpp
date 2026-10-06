#pragma once
// The kernels of the batched grounded heuristics (cuda/heuristics.hpp): h_max, h_add and
// h_FF of B states per launch over the relaxed grounding of heuristics/relaxed_task.hpp, one group of threads per state
// (a block per state; for small groundings a warp per state, several states per block), the groups looping
// over the states of a launch. Destination-passing launchers over the arrays the host driver owns.
//
// Per state:
//   1. the state's fluent bits become propositions (Relaxed::slot_pos / slot_neg): "p true" costs 0, "p false" costs 0
//      unless p holds, everything else starts at infinity;
//   2. costs, the least fixpoint of cost(e) = min(cost(e), min over the operators o adding e of c(o) + max / sum of
//      cost(pre(o))): the values of heuristics::make_heuristic (a generalized Dijkstra) exactly, saturating sums
//      included. Two variants:
//        k_sweep     Jacobi rounds of that update: every round evaluates every operator on the previous round's costs
//                    and lowers a second array with atomicMin, until no cost changes (rounds x operators work);
//        k_frontier  the bucketed frontier: per operator a counter of its unsettled precondition entries; a
//                    round settles every unsettled proposition of the cheapest cost and decrements the counters of the
//                    operators it is a precondition of (Relaxed::pre_of_lanes lanes per proposition); an operator whose
//                    counter reaches 0 fires once and lowers its effects and the next round's cost with atomicMin.
//                    Every operator is evaluated once; h_max and h_add stop when every goal is settled, h_FF when the
//                    next bucket is dearer than every goal.
//      A round's result does not depend on the order of the threads in either;
//   3. h_FF (on the h_max costs): best supporters by BFS level over the tight achievers (an operator o is tight for
//      e in eff(o) when c(o) + max cost(pre(o)) = cost(e)). Level 0 are the propositions that cost 0 from the start;
//      a proposition gets level r in round r from the tight achievers whose preconditions have levels at most r - 1,
//      one of them r - 1; its supporter is the smallest such operator id (atomicMin). The sweeps evaluate every
//      operator per level, the frontier each operator once, when its last precondition gets its level (counters). An
//      axiom passes on the supporter of its precondition of level r - 1 with the smallest proposition id (mimir's
//      "derived atom takes over the supporter of its axiom's last settled precondition"). The relaxed plan is the
//      closure of the goal under x -> pre(supp(x)), counted once per ground action; the closure is a set, so the order
//      of the frontier does not matter. heuristics::make_heuristic breaks the ties among supporters by its queue order
//      instead, so h_FF equals the CPU's where no ties decide; heuristics.hpp has the CPU reference of this rule.
//      Uniform costs (Relaxed::uniform_cost: every operator an action of the same cost c > 0) skip the levels:
//      there the level of a proposition is its cost / c (by induction over the costs: the preconditions of a tight
//      achiever of e cost at most cost(e) - c, one of them exactly), so every tight achiever of e becomes ready in the
//      same round and the supporter is the smallest tight achiever of e, which the relaxed plan looks up for the
//      propositions it reaches (Relaxed::ach, ascending ids; the first tight one). Same values, no levels phase.
// The scratch of a group (costs, levels, lists, the operators' counters, the bitsets of the relaxed plan's
// propositions and ground actions) lives in dynamic shared memory when it fits, otherwise in global memory (one slice
// per group). Nothing in it carries over from one state to the next.
//
// Atomics decide list slots (the order of a frontier) and minima (atomicMin of costs and of operator ids), never an
// order that reaches the result: every phase is a set operation or a minimum.
//
// Device-code subset: included by .cu files compiled by nvcc as C++20.

#include "mymyr/core/types.hpp"

#include <cuda_runtime_api.h>

namespace mymyr::cuda::hk
{
inline constexpr u32 k_inf = 0xFFFFFFFFu;   // infinite relaxed cost; as an h value: a dead end
inline constexpr u32 k_none = 0xFFFFFFFFu;  // no operator / proposition

/// Heuristic kinds of a launch.
enum : u32
{
    k_max = 0,
    k_add = 1,
    k_ff = 2,
};

/// Cost variants of a launch.
enum : u32
{
    k_sweep = 0,
    k_frontier = 1,
};

/// Per-state status of a launch.
enum : u8
{
    k_ok = 0,
    k_outside = 1,  // an atom of the state has no proposition (outside the grounding): the host evaluates it
    k_invalid = 2,  // the state sets a slot past Relaxed::slots
};

/// The relaxed grounding on the device (heuristics::RelaxedTask, flat CSR).
struct Relaxed
{
    u32 P = 0;   // propositions
    u32 O = 0;   // operators
    u32 G = 0;   // goal propositions
    u32 GA = 0;  // ground actions
    const u32* pre_begin = nullptr;  // [O + 1]
    const u32* pre = nullptr;        // preconditions, with multiplicity (h_add counts a repeated one twice)
    const u32* eff_begin = nullptr;  // [O + 1]
    const u32* eff = nullptr;
    const u32* opcost = nullptr;     // [O]: unit or real action cost, 0 for axioms
    const u8* axiom = nullptr;       // [O]
    const u32* op_ga = nullptr;      // [O]: ground action (k_none for axioms)
    const u32* ga_cost = nullptr;    // [GA]: what h_FF counts per ground action (1, or its real cost)
    const u32* pre_of_begin = nullptr;  // [P + 1]
    const u32* pre_of = nullptr;        // operators with p in their preconditions (with multiplicity)
    u32 pre_of_lanes = 1;               // the frontier's lanes per settled proposition (1..32, a power of two: the
                                        // mean length of the pre_of lists, rounded up)
    const u32* ach_begin = nullptr;     // [P + 1]
    const u32* ach = nullptr;           // operators with p among their effects, ascending
    u32 uniform_cost = 0;               // c > 0 when every operator is an action of cost c (no axioms): h_FF's
                                        // supporters are the smallest tight achievers (no levels; see above), 0 else
    u32 ach_lanes = 1;                  // lanes per proposition of the supporter lookup (1..32, a power of two: the
                                        // mean length of the ach lists, as pre_of_lanes)
    const u32* zero_ops = nullptr;      // operators without "true" preconditions
    u32 num_zero = 0;
    const u32* goal = nullptr;   // [G] distinct
    const u8* is_goal = nullptr;  // [P]
    const u8* negative = nullptr;  // [P]: a "false" proposition
    const u32* slot_pos = nullptr;  // [slots]: "true" proposition of a fluent slot (k_none: outside the grounding)
    const u32* slot_neg = nullptr;  // [slots]: "false" proposition (k_none: none)
    u32 slots = 0;
    u32 goal_unreachable = 0;
};

/// Rows of fluent words (row i at data + i * stride).
struct Rows
{
    const u64* data = nullptr;
    u64 stride = 0;
    u32 words = 0;
    u64 n = 0;
    const u32* count = nullptr;  // a device count: the first min(n, *count) rows (null: n)
};

/// A launch configuration. No result depends on it (group size, threads, blocks, shared or global scratch).
struct Launch
{
    u32 kind = k_ff;
    u32 variant = k_frontier;
    u32 threads = 128;  // per block, a multiple of 32
    u32 blocks = 1;     // grid (the groups loop over the states)
    u32 warp = 0;       // 1: every warp of a block evaluates its own states (small groundings); 0: the whole block
    u32 shared = 1;     // 1: scratch in dynamic shared memory; 0: global slices
    void* scratch = nullptr;  // global mode: groups * group_bytes
    u64 group_bytes = 0;      // scratch bytes per group (scratch_bytes)
};

/// Groups of a launch (the scratch slices of global mode).
[[nodiscard]] MYMYR_HD u64 groups_of(const Launch& l) { return u64{l.blocks} * (l.warp ? l.threads / 32 : 1); }

/// Outputs: h[i] (k_inf: dead end or not evaluated), status[i] (k_ok, k_outside, k_invalid) and, in counters[0] /
/// counters[1], the numbers of states with status k_outside / k_invalid (added to).
struct Out
{
    u32* h = nullptr;
    u8* status = nullptr;
    u32* counters = nullptr;
};

/// Scratch bytes of one group (the layout the kernels carve out of it; a multiple of 16). uniform: Relaxed::uniform_cost
/// is set (h_FF keeps no levels).
[[nodiscard]] MYMYR_HD u64 scratch_bytes(u32 P, u32 O, u32 GA, u32 kind, u32 variant, bool uniform)
{
    u64 w = 2 * u64{P};  // cost, nxt (sweeps: the next round's costs; frontier: the settled flags)
    w += P;              // list T (settled / newly leveled propositions, extraction frontier)
    if (variant == k_frontier)
        w += u64{P} + O;  // the second level list, the operators' counters
    if (kind == k_ff)
    {
        if (!uniform)
            w += 3 * u64{P};                            // level, candidate supporter, supporter
        w += (u64{P} + 31) / 32 + (u64{GA} + 31) / 32;  // relaxed-plan bits: propositions, ground actions
    }
    return (w * sizeof(u32) + 15) / 16 * 16;
}

/// h of rows.n states (out.h[i] for row i). l.scratch (global mode) must not be used by another launch at the same
/// time.
cudaError_t launch_evaluate(const Relaxed& r, Rows rows, Launch l, Out out, cudaStream_t s);

/// out[i] = h[i] as a double (+inf for k_inf).
cudaError_t launch_to_f64(const u32* h, u64 n, f64* out, cudaStream_t s);
/// Resident blocks per SM of the launch's kernel with its dynamic shared memory (0 when it does not fit).
cudaError_t occupancy(const Launch& l, int* blocks_per_sm);
}  // namespace mymyr::cuda::hk
