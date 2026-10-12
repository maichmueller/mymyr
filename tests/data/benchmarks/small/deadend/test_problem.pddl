;; `armed` is absent from the initial state and no action adds it, so nothing is applicable and
;; the goal is unreachable.
(define (problem deadend-initial)
 (:domain deadend)
 (:objects)
 (:init)
 (:goal (and (done)))
)
