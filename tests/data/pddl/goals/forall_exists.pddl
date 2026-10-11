(define (problem forall-exists) (:domain goals)
  (:objects a b - place
            t1 t2 - thing
            b1 - big)
  (:init (road a b) (road b a) (heavy t1) (heavy b1))
  (:goal (and (lit a) (forall (?p - place) (exists (?t - thing) (at ?t ?p))))))
