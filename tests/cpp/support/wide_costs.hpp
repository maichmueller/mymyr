#pragma once
// Test support: a PDDL domain whose action costs need the device's cost programs (cuda/cost_program.hpp):
// five cost parameters, a static function of 30^5 > 2^24 keys, operands evaluated in both orders (the deeper first:
// reversed - and /). The device state spaces and A* compare their costs with the CPU's on it.

#include "mymyr/core/types.hpp"

#include <string>
#include <utility>

namespace mymyr::test
{
/// The domain; `hop` is the hop action's cost expression (over ?b).
inline std::string wide_domain(const std::string& hop)
{
    return R"(
(define (domain wide)
 (:requirements :strips :typing :action-costs)
 (:types cell)
 (:predicates (at ?c - cell) (link ?a ?b ?c ?d ?e - cell) (hop ?a ?b - cell))
 (:functions (total-cost) (w5 ?a ?b ?c ?d ?e - cell) (w1 ?a - cell))
 (:action go :parameters (?a ?b ?c ?d ?e - cell) :precondition (and (at ?a) (link ?a ?b ?c ?d ?e))
   :effect (and (not (at ?a)) (at ?b) (increase (total-cost) (+ (w5 ?a ?b ?c ?d ?e) (* 2 (- (w1 ?c) (w1 ?d)))))))
 (:action hop :parameters (?a ?b - cell) :precondition (and (at ?a) (hop ?a ?b))
   :effect (and (not (at ?a)) (at ?b) (increase (total-cost) )" +
           hop + R"()))
)
)";
}

/// Integral hop costs (a reversed subtraction), real ones (a reversed division), negative ones for w1 > 5.
inline const std::string k_wide_integral = "(- 20 (* (w1 ?b) 1))";
inline const std::string k_wide_real = "(/ 100 (+ (w1 ?b) 3))";
inline const std::string k_wide_negative = "(- 5 (w1 ?b))";

/// A ring of n cells: `go` both ways (five-object links, w5 from 100 to 149), `hop` two cells ahead (w1 = i mod 10);
/// the goal is the cell opposite c0.
inline std::string wide_problem(u32 n)
{
    const auto c = [](u32 i) { return " c" + std::to_string(i); };
    std::string p = "(define (problem p) (:domain wide)\n (:objects";
    for (u32 i = 0; i < n; ++i)
        p += c(i);
    p += " - cell)\n (:init (at c0) (= (total-cost) 0)";
    for (u32 i = 0; i < n; ++i)
    {
        const u32 j = (i + 1) % n;
        for (const auto& [a, b] : {std::pair{i, j}, std::pair{j, i}})
        {
            const std::string args = c(a) + c(b) + c((a * 7 + 3) % n) + c((b * 11 + 5) % n) + c((a * 13 + b) % n);
            p += " (link" + args + ") (= (w5" + args + ") " + std::to_string(100 + (a * 17 + b) % 50) + ")";
        }
        p += " (hop" + c(i) + c((i + 2) % n) + ") (= (w1" + c(i) + ") " + std::to_string(i % 10) + ")";
    }
    p += ")\n (:goal (at c" + std::to_string(n / 2) + "))\n (:metric minimize (total-cost)))\n";
    return p;
}
}  // namespace mymyr::test
