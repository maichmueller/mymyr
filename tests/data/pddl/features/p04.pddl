(define (problem features-p04) (:domain features) (:objects a b c)
 (:init (fixed a) (flag a))
 (:goal (and (not (ready)) (fixed b) (flag b))))
