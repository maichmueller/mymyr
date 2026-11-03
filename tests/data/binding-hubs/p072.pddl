; agents = 4, jobs = 4, hubs = 8
(define (problem p072) (:domain binding-hubs)
  (:objects a12 a34 a38 a39 - agent
            t1 t4 t18 t22 t56 t57 t60 t62 t68 t86 t89 t90 t96 t101 t102 t105 - tool
            h7 h43 h46 h48 h59 h67 h68 h75 - hub
            x12 x13 x16 x27 - job)
  (:init (end t1 h48) (end t1 h68) (end t101 h43) (end t101 h7) (end t102 h43) (end t102 h75) (end t105 h46) (end t105 h68) (end t18 h46) (end t18 h59) (end t22 h7) (end t22 h75) (end t4 h67) (end t4 h75) (end t56 h68) (end t56 h7) (end t57 h46) (end t57 h75) (end t60 h48) (end t60 h67) (end t62 h46) (end t62 h48) (end t68 h59) (end t68 h68) (end t86 h67) (end t86 h7) (end t89 h48) (end t89 h59) (end t90 h43) (end t90 h59) (end t96 h43) (end t96 h67) (free t1) (free t101) (free t102) (free t105) (free t18) (free t22) (free t4) (free t56) (free t57) (free t60) (free t62) (free t68) (free t86) (free t89) (free t90) (free t96) (link h43 h67) (link h43 h68) (link h46 h59) (link h46 h67) (link h48 h68) (link h48 h75) (link h59 h46) (link h59 h7) (link h67 h43) (link h67 h46) (link h68 h43) (link h68 h48) (link h7 h59) (link h7 h75) (link h75 h48) (link h75 h7) (open x12) (open x13) (open x16) (open x27) (ready a12) (ready a34) (ready a38) (ready a39))
  (:goal (and (done x12) (done x13) (done x16) (done x27))))
