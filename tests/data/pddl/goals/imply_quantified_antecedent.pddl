(define (problem imply-quantified-antecedent) (:domain goals)
  (:objects a b - place
            t1 t2 - thing
            b1 - big)
  (:init (road a b) (road b a) (heavy t1) (heavy b1))
  (:goal (and (imply (exists (?t - thing) (heavy ?t)) (flag)) (imply (forall (?t - big) (heavy ?t)) (ok t2)) (imply (forall (?t - thing) (heavy ?t)) (lit b)))))
