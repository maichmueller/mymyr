(define (problem exists-forall) (:domain goals)
  (:objects a b - place
            t1 t2 - thing
            b1 - big)
  (:init (road a b) (road b a) (heavy t1) (heavy b1))
  (:goal (and (lit a) (exists (?t - thing) (forall (?p - place) (imply (road ?p a) (at ?t ?p)))))))
