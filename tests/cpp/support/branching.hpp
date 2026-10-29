#pragma once
// Test support: a PDDL task with a very wide layer. Layer 1 holds `modes` states, each with n^3 successors that are
// distinct and have no successors of their own: 1 + modes + modes * n^3 states and modes + modes * n^3 transitions,
// one goal state at depth 2. One expansion (modes 1) or a few concurrent ones (modes 8) insert more states than a
// concurrent store's initial table holds once n^3 > 2^18 or modes * n^3 > 2^18.

#include "mymyr/core/types.hpp"

#include <string>

namespace mymyr::test
{
inline std::string branching_domain()
{
    return R"(
(define (domain branching)
 (:requirements :strips :typing)
 (:types mode obj)
 (:predicates (start) (on ?m - mode) (done ?m - mode) (a ?x - obj) (b ?x - obj) (c ?x - obj))
 (:action choose :parameters (?m - mode) :precondition (start) :effect (and (not (start)) (on ?m)))
 (:action pick :parameters (?m - mode ?x ?y ?z - obj) :precondition (on ?m)
   :effect (and (not (on ?m)) (done ?m) (a ?x) (b ?y) (c ?z)))
)
)";
}

inline std::string branching_problem(u32 modes, u32 n)
{
    std::string p = "(define (problem p) (:domain branching) (:objects";
    for (u32 m = 1; m <= modes; ++m)
        p += " m" + std::to_string(m);
    p += " - mode";
    for (u32 i = 1; i <= n; ++i)
        p += " o" + std::to_string(i);
    p += " - obj) (:init (start)) (:goal (and (done m1) (a o1) (b o1) (c o1))))";
    return p;
}

inline u64 branching_states(u32 modes, u32 n) { return 1 + modes + u64{modes} * n * n * n; }
}  // namespace mymyr::test
