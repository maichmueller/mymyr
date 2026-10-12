;; The same domain with `armed` in the initial state: exactly one applicable action, so the
;; existence test and the enumeration have something to agree on.
(define (problem deadend-live)
 (:domain deadend)
 (:objects)
 (:init (armed))
 (:goal (and (done)))
)
