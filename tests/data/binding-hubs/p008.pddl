; agents = 2, jobs = 1, hubs = 8
(define (problem p008) (:domain binding-hubs)
  (:objects a3 a11 - agent
            t5 t7 t9 t11 t35 t49 t61 t69 t72 t80 t91 t104 t117 t145 t147 t156 - tool
            h7 h10 h14 h15 h19 h48 h58 h69 - hub
            x4 - job)
  (:init (end t104 h48) (end t104 h7) (end t11 h10) (end t11 h48) (end t117 h10) (end t117 h15) (end t145 h48) (end t145 h58) (end t147 h19) (end t147 h58) (end t156 h14) (end t156 h69) (end t35 h19) (end t35 h69) (end t49 h14) (end t49 h19) (end t5 h10) (end t5 h14) (end t61 h10) (end t61 h58) (end t69 h15) (end t69 h48) (end t7 h69) (end t7 h7) (end t72 h15) (end t72 h58) (end t80 h15) (end t80 h69) (end t9 h19) (end t9 h7) (end t91 h14) (end t91 h7) (free t104) (free t11) (free t117) (free t145) (free t147) (free t156) (free t35) (free t49) (free t5) (free t61) (free t69) (free t7) (free t72) (free t80) (free t9) (free t91) (link h10 h19) (link h10 h48) (link h14 h58) (link h14 h69) (link h15 h58) (link h15 h7) (link h19 h10) (link h19 h7) (link h48 h10) (link h48 h69) (link h58 h14) (link h58 h15) (link h69 h14) (link h69 h48) (link h7 h15) (link h7 h19) (open x4) (ready a11) (ready a3))
  (:goal (and (done x4))))
