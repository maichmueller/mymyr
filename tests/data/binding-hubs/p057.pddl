; agents = 4, jobs = 3, hubs = 10
(define (problem p057) (:domain binding-hubs)
  (:objects a3 a35 a37 a39 - agent
            t9 t12 t14 t39 t66 t73 t81 t84 t85 t115 t116 t120 t132 t136 t138 t152 t153 t160 t161 t189 - tool
            h7 h15 h17 h25 h52 h61 h63 h71 h95 h97 - hub
            x6 x8 x27 - job)
  (:init (end t115 h25) (end t115 h71) (end t116 h15) (end t116 h63) (end t12 h17) (end t12 h71) (end t120 h15) (end t120 h61) (end t132 h25) (end t132 h95) (end t136 h52) (end t136 h7) (end t138 h63) (end t138 h95) (end t14 h63) (end t14 h7) (end t152 h61) (end t152 h7) (end t153 h52) (end t153 h97) (end t160 h63) (end t160 h71) (end t161 h17) (end t161 h97) (end t189 h61) (end t189 h97) (end t39 h25) (end t39 h97) (end t66 h17) (end t66 h95) (end t73 h52) (end t73 h95) (end t81 h15) (end t81 h52) (end t84 h15) (end t84 h17) (end t85 h25) (end t85 h7) (end t9 h61) (end t9 h71) (free t115) (free t116) (free t12) (free t120) (free t132) (free t136) (free t138) (free t14) (free t152) (free t153) (free t160) (free t161) (free t189) (free t39) (free t66) (free t73) (free t81) (free t84) (free t85) (free t9) (link h15 h25) (link h15 h63) (link h17 h7) (link h17 h97) (link h25 h15) (link h25 h95) (link h52 h7) (link h52 h71) (link h61 h71) (link h61 h95) (link h63 h15) (link h63 h97) (link h7 h17) (link h7 h52) (link h71 h52) (link h71 h61) (link h95 h25) (link h95 h61) (link h97 h17) (link h97 h63) (open x27) (open x6) (open x8) (ready a3) (ready a35) (ready a37) (ready a39))
  (:goal (and (done x27) (done x6) (done x8))))
