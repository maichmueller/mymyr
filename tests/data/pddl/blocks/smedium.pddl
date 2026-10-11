
(define (problem blocks-3-0)
    (:domain blocks)

    (:objects b a c )

    (:init (clear a)
           (on a b)
           (on b c)
           (ontable c)
           (handempty))

    (:goal (and (on c b)
                (on b a)))
    )
