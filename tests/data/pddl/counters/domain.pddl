; Bounded numeric counters for state-space parity checks on numeric tasks: two counters, moves with a numeric
; precondition, a numeric action cost, and a flag set when both counters meet a target.
(define (domain counters)
 (:requirements :strips :typing :numeric-fluents :negative-preconditions :action-costs)
 (:types counter)
 (:predicates (done) (locked ?c - counter))
 (:functions (value ?c - counter) (limit) (step ?c - counter) (total-cost))
 (:action inc
   :parameters (?c - counter)
   :precondition (and (not (locked ?c)) (<= (+ (value ?c) (step ?c)) (limit)))
   :effect (and (increase (value ?c) (step ?c)) (increase (total-cost) (step ?c))))
 (:action dec
   :parameters (?c - counter)
   :precondition (and (not (locked ?c)) (>= (value ?c) (step ?c)))
   :effect (and (decrease (value ?c) (step ?c)) (increase (total-cost) 1)))
 (:action lock
   :parameters (?c - counter)
   :precondition (and (not (locked ?c)) (= (value ?c) (limit)))
   :effect (and (locked ?c) (increase (total-cost) 2)))
 (:action finish
   :parameters (?a ?b - counter)
   :precondition (and (locked ?a) (locked ?b) (not (done)))
   :effect (and (done)))
)
