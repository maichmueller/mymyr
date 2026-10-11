(define (problem features-p02) (:domain features) (:objects a b c)
 (:init (fixed a) (flag a))
 (:goal (and (forall (?x) (flag ?x)) (ready))))
