(define (problem features-p03) (:domain features) (:objects a b c)
 (:init (fixed a) (flag a))
 (:goal (and (imply (fixed a) (flag c)) (imply (fixed b) (flag b)) (not (flag a)))))
