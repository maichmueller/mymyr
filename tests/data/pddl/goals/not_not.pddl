(define (problem not-not) (:domain goals)
  (:objects a b - place
            t1 t2 - thing
            b1 - big)
  (:init (road a b) (road b a) (heavy t1) (heavy b1))
  (:goal (and (not (not (lit a))) (not (not (not (ok t1)))))))
