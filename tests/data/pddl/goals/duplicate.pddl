(define (problem duplicate) (:domain goals)
  (:objects a b - place
            t1 t2 - thing
            b1 - big)
  (:init (road a b) (road b a) (heavy t1) (heavy b1))
  (:goal (and (lit a) (lit a) (forall (?p - place) (lit ?p)) (not (ok t1)) (not (ok t1)))))
