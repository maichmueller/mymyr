(define (problem features-p02-expanded) (:domain features) (:objects a b c)
 (:init (fixed a) (flag a))
 (:goal (and (flag a) (flag b) (flag c) (ready))))
