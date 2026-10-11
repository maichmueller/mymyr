(define (problem imply-equality) (:domain goals)
  (:objects a b - place
            t1 t2 - thing
            b1 - big)
  (:init (road a b) (road b a) (heavy t1) (heavy b1))
  (:goal (and (imply (= a b) (lit b)) (imply (not (= a b)) (lit a)))))
