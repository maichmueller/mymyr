(define (problem exists-single-witness) (:domain goals)
  (:objects a b - place
            t1 t2 - thing
            b1 - big)
  (:init (road a b) (road b a) (heavy t1) (heavy b1))
  (:goal (exists (?p - place) (and (road ?p a) (lit ?p)))))
