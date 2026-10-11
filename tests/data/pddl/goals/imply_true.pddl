(define (problem imply-true) (:domain goals)
  (:objects a b - place
            t1 t2 - thing
            b1 - big)
  (:init (road a b) (road b a) (heavy t1) (heavy b1))
  (:goal (and (imply (heavy t1) (ok t1)) (imply (heavy b1) (forall (?p - place) (at b1 ?p))))))
