(define (problem spanner-iw1-incremental-regression)
 (:domain spanner)
 (:objects
    bob - man
    spanner1 spanner2 spanner3 spanner4 spanner5 - spanner
    nut1 - nut
    gate - location)
 (:init
    (at bob gate)
    (at nut1 gate)
    (loose nut1)
    (carrying bob spanner1)
    (useable spanner1)
    (carrying bob spanner2)
    (useable spanner2)
    (carrying bob spanner3)
    (useable spanner3)
    (carrying bob spanner4)
    (useable spanner4)
    (carrying bob spanner5)
    (useable spanner5))
 (:goal
    (and
        (tightened nut1))))
