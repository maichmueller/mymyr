(define (domain landmark-cond-effect-dedup)
  (:requirements :strips :conditional-effects)
  (:predicates (p) (s1) (s2) (r))

  (:action init-s
    :parameters ()
    :precondition (p)
    :effect (and (s1) (s2) (p)))

  (:action act
    :parameters ()
    :precondition (p)
    :effect (and
      (when (s1) (r))
      (when (s2) (r)))))
