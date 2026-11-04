; Gadgets in a chain, each with cells: linking a gadget's cell needs a road to every linked cell of the gadgets it
; borders (a universal precondition with an implication), and a cell can be unlinked when no linked cell is next to it.
(define (domain linking)
  (:requirements :strips :typing :negative-preconditions :universal-preconditions :existential-preconditions
                 :disjunctive-preconditions)
  (:types gadget cell)
  (:predicates (in ?c - cell ?g - gadget) (road ?c - cell ?d - cell) (border ?g - gadget ?u - gadget)
               (next ?g - gadget ?h - gadget) (terminal ?g - gadget) (cur ?g - gadget) (linked ?c - cell) (done))
  (:action link
    :parameters (?g - gadget ?h - gadget ?c - cell)
    :precondition (and (cur ?g) (next ?g ?h) (in ?c ?g)
                       (forall (?u - gadget ?p - cell) (imply (and (border ?g ?u) (in ?p ?u) (linked ?p)) (road ?c ?p))))
    :effect (and (not (cur ?g)) (cur ?h) (linked ?c)))
  (:action unlink
    :parameters (?c - cell)
    :precondition (and (linked ?c) (forall (?d - cell) (imply (road ?c ?d) (not (linked ?d)))))
    :effect (not (linked ?c)))
  (:action back
    :parameters (?g - gadget ?h - gadget)
    :precondition (and (cur ?h) (next ?g ?h))
    :effect (and (not (cur ?h)) (cur ?g)))
  (:action finish
    :parameters (?g - gadget)
    :precondition (and (cur ?g) (terminal ?g))
    :effect (and (not (cur ?g)) (done))))
