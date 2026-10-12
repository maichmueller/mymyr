;; A conditional add whose condition is itself only reachable through another action.
;;
;; `open` adds `unlocked(?g)` only when `charged(?g)` holds, and `charged(?g)` is produced by `charge`,
;; which needs the static `chargeable(?g)`. An engine that treated the conditional effect as unconditional
;; would reach `unlocked(g2)`; one that dropped conditional effects would reach neither.
(define (domain relaxed-reachability-chain)
  (:requirements :strips :typing :conditional-effects)
  (:types gadget)
  (:predicates
    (powered ?g - gadget)
    (chargeable ?g - gadget)
    (charged ?g - gadget)
    (unlocked ?g - gadget))

  (:action charge
    :parameters (?g - gadget)
    :precondition (and (powered ?g) (chargeable ?g))
    :effect (charged ?g))

  (:action open
    :parameters (?g - gadget)
    :precondition (powered ?g)
    :effect (when (charged ?g) (unlocked ?g)))
)
