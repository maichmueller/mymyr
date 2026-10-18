(define (problem wide-predicates-p01) (:domain wide-predicates)
  (:objects a b c - obj)
  (:init (link a b c) (link c b a) (on a))
  (:goal (and (link a b c) (not (link b b b)) (trio a b c) (not (trio b c a)) (on b))))
