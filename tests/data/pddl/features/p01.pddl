(define (problem features-p01) (:domain features) (:objects a b)
 (:init (fixed a) (flag a))
 (:goal (and (flag b) (not (flag a)) (fixed a) (not (fixed b))
             (available b) (not (available a)) (ready))))
