; agents = 3, jobs = 2, hubs = 8
(define (problem p032) (:domain binding-hubs)
  (:objects a11 a14 a30 - agent
            t4 t5 t28 t49 t63 t67 t79 t82 t85 t88 t103 t113 t136 t156 t157 t160 - tool
            h14 h21 h27 h38 h43 h49 h66 h72 - hub
            x9 x18 - job)
  (:init (end t103 h38) (end t103 h43) (end t113 h14) (end t113 h27) (end t136 h14) (end t136 h66) (end t156 h27) (end t156 h38) (end t157 h14) (end t157 h21) (end t160 h49) (end t160 h72) (end t28 h21) (end t28 h66) (end t4 h27) (end t4 h66) (end t49 h21) (end t49 h27) (end t5 h21) (end t5 h43) (end t63 h43) (end t63 h49) (end t67 h38) (end t67 h72) (end t79 h38) (end t79 h49) (end t82 h66) (end t82 h72) (end t85 h14) (end t85 h49) (end t88 h43) (end t88 h72) (free t103) (free t113) (free t136) (free t156) (free t157) (free t160) (free t28) (free t4) (free t49) (free t5) (free t63) (free t67) (free t79) (free t82) (free t85) (free t88) (link h14 h21) (link h14 h38) (link h21 h14) (link h21 h72) (link h27 h49) (link h27 h66) (link h38 h14) (link h38 h43) (link h43 h38) (link h43 h66) (link h49 h27) (link h49 h72) (link h66 h27) (link h66 h43) (link h72 h21) (link h72 h49) (open x18) (open x9) (ready a11) (ready a14) (ready a30))
  (:goal (and (done x18) (done x9))))
