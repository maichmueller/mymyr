;; Conditional effects as achievers, in the two shapes `landmark_cond_effect_dedup` cannot express.
;;
;; `target` has exactly one achiever, whose add is *inside* a `when`: its effect condition `guard`
;; has to join the precondition intersection, or `guard` is lost.
;; `other` has two achievers in two different actions -- one unconditional, one conditional -- whose
;; preconditions differ. If the conditional add were not counted as an achiever of its own, the
;; intersection would run over `add-other-plain` alone and wrongly report `shared` as a landmark.
(define (domain landmark-lifted-cond-effect)
  (:requirements :strips :conditional-effects)
  (:predicates (base) (guard) (shared) (extra) (target) (other))

  (:action gated
    :parameters ()
    :precondition (base)
    :effect (when (guard) (target)))

  (:action add-other-plain
    :parameters ()
    :precondition (and (base) (shared))
    :effect (and (other)))

  (:action add-other-gated
    :parameters ()
    :precondition (base)
    :effect (when (extra) (other)))

  ;; Present only so that every predicate above appears in an add effect and is classified fluent.
  (:action bootstrap
    :parameters ()
    :precondition (and)
    :effect (and (base) (guard) (shared) (extra))))
