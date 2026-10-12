(define (problem forall-partly-expandable) (:domain goals)
  (:objects a b - place
            t1 t2 - thing
            b1 - big)
  (:init (road a b) (road b a) (heavy t1) (heavy b1))
  (:goal (forall (?t - big) (and (imply (heavy ?t) (ok ?t)) (or (seen ?t) (flag))))))
