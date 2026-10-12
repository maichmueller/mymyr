;; A domain whose only action can never fire in a state where `armed` is false, so a problem that
;; starts without it has an initial state with no applicable action at all. It exists so the
;; dead-end answer of `has_applicable_action` has a fixture that is unambiguously a dead end
;; rather than a state that merely happens to be one at the end of a long search.
(define (domain deadend)
  (:requirements :strips)
  (:predicates (armed) (done))

  (:action fire
    :parameters ()
    :precondition (and (armed))
    :effect (and (done)))
)
