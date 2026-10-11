(define (problem negated-forall-single-witness) (:domain goals)
  (:objects a b - place
            t1 t2 - thing
            b1 - big)
  (:init (road a b) (road b a) (heavy t1) (heavy b1))
  (:goal (not (forall (?t - thing) (or (not (heavy ?t)) (= ?t t1) (ok ?t))))))
