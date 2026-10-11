(define (problem forall-nested) (:domain goals)
  (:objects a b - place
            t1 t2 - thing
            b1 - big)
  (:init (road a b) (road b a) (heavy t1) (heavy b1))
  (:goal (forall (?t - big) (forall (?p - place) (at ?t ?p)))))
