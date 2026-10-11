(define (problem features-p03-expanded) (:domain features) (:objects a b c)
 (:init (fixed a) (flag a))
 (:goal (and (flag c) (not (flag a)))))
