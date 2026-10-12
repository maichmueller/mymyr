;; A recursive derived predicate: `above` is the transitive closure of the fluent `on`.
;;
;; The axiom is a genuine rule of the fixpoint, so the engine has to iterate it -- one pass over the axioms
;; reaches `above(b1, b2)` but not `above(b1, b3)`.
(define (domain relaxed-reachability-axiom)
  (:requirements :strips :typing :derived-predicates :existential-preconditions)
  (:types block)
  (:predicates
    (on ?x - block ?y - block)
    (stackable ?x - block ?y - block)
    (clear ?x - block)
    (above ?x - block ?y - block))

  (:derived (above ?x - block ?y - block)
    (on ?x ?y))

  (:derived (above ?x - block ?y - block)
    (exists (?z - block) (and (on ?x ?z) (above ?z ?y))))

  (:action stack
    :parameters (?x - block ?y - block)
    :precondition (and (stackable ?x ?y) (clear ?y))
    :effect (and (on ?x ?y) (not (clear ?y))))
)
