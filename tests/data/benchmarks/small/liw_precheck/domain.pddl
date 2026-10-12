;; Fixtures for the LIW(1) action-level novelty precheck. Every action exists to put the precheck
;; in a position where a plausible-but-wrong implementation answers differently than testing the
;; successor itself would.
;;
;; `lm1` is the landmark in every scenario; `r` is an ordinary atom that can be introduced late, so
;; a pair `(lm1, {r})` can be made the only unseen one.
(define (domain liw-precheck)
  (:requirements :strips :negative-preconditions)
  (:predicates (p) (q) (r) (lm1) (lm2) (done))

  ;; An atom in BOTH the positive and the negative effect list. `apply_action_effects` subtracts
  ;; the negatives and then unions the positives, so `p` survives -- but a delete list built by
  ;; "true in s" alone reports it as deleted and the reconstruction loses it.
  (:action both
    :parameters ()
    :precondition (and (p) (not (done)))
    :effect (and (not (p)) (p) (done)))

  ;; Switches the landmark on while adding nothing else at all.
  (:action flip-lm1
    :parameters ()
    :precondition (and (p) (not (lm1)))
    :effect (and (lm1)))

  ;; Switches the landmark on AND adds an ordinary atom in the same step. Run after `lm1` has
  ;; already been paired with every atom of the predecessor, the pair `(lm1, {r})` is the only
  ;; unseen one -- and it exists only in the successor's atoms.
  (:action flip-lm1-and-add-r
    :parameters ()
    :precondition (and (p) (not (lm1)) (not (r)))
    :effect (and (lm1) (r)))

  ;; Deletes the last true landmark and adds nothing, so the `BOT` coordinate flips on with an
  ;; empty `Add & L`.
  (:action drop-lm1
    :parameters ()
    :precondition (and (lm1))
    :effect (and (not (lm1))))

  ;; Introduces the ordinary atom without touching any landmark.
  (:action add-r
    :parameters ()
    :precondition (and (p) (not (r)))
    :effect (and (r)))

  ;; A plain step over ordinary atoms: the common case the fast path is built for.
  (:action ordinary
    :parameters ()
    :precondition (and (p) (not (q)))
    :effect (and (q))))
