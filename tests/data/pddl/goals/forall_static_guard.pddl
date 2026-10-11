(define (problem forall-static-guard) (:domain goals)
  (:objects a b - place
            t1 t2 - thing
            b1 - big)
  (:init (road a b) (road b a) (heavy t1) (heavy b1))
  (:goal (forall (?t - thing) (imply (heavy ?t) (ok ?t)))))
