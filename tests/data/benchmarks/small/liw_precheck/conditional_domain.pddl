;; The same fixtures written with conditional effects, so the precheck's general branch (resolve
;; the triggered effects against the state) can be compared against the cached-effect fast path on
;; behaviour that is meant to be identical.
;;
;; Every condition holds whenever the action is applicable, so the two domains induce the same
;; transition system -- only how the precheck derives the transition differs.
(define (domain liw-precheck-conditional)
  (:requirements :strips :negative-preconditions :conditional-effects)
  (:predicates (p) (q) (r) (lm1) (lm2) (done))

  (:action both
    :parameters ()
    :precondition (and (p) (not (done)))
    :effect (and (when (p) (and (not (p)) (p) (done)))))

  (:action flip-lm1
    :parameters ()
    :precondition (and (p) (not (lm1)))
    :effect (and (when (p) (lm1))))

  (:action flip-lm1-and-add-r
    :parameters ()
    :precondition (and (p) (not (lm1)) (not (r)))
    :effect (and (when (p) (and (lm1) (r)))))

  (:action drop-lm1
    :parameters ()
    :precondition (and (lm1))
    :effect (and (when (lm1) (not (lm1)))))

  (:action add-r
    :parameters ()
    :precondition (and (p) (not (r)))
    :effect (and (when (p) (r))))

  (:action ordinary
    :parameters ()
    :precondition (and (p) (not (q)))
    :effect (and (when (p) (q)))))
