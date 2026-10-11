(define (problem blocks-flags-p01) (:domain blocks-flags)
 (:objects a b c)
 (:init (clear a) (on a b) (on b c) (ontable c) (handempty))
 (:goal (and (on c b) (on b a) (moved) (stacked))))
