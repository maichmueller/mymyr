; agents = 4, jobs = 1, hubs = 10
(define (problem p044) (:domain binding-hubs)
  (:objects a4 a7 a10 a24 - agent
            t6 t15 t22 t31 t36 t62 t66 t83 t84 t93 t98 t99 t116 t133 t172 t186 t187 t197 t198 t199 - tool
            h6 h20 h25 h31 h39 h51 h63 h66 h77 h78 - hub
            x3 - job)
  (:init (end t116 h25) (end t116 h77) (end t133 h20) (end t133 h78) (end t15 h63) (end t15 h77) (end t172 h31) (end t172 h66) (end t186 h51) (end t186 h63) (end t187 h39) (end t187 h51) (end t197 h66) (end t197 h77) (end t198 h25) (end t198 h51) (end t199 h31) (end t199 h78) (end t22 h20) (end t22 h39) (end t31 h6) (end t31 h78) (end t36 h20) (end t36 h25) (end t6 h25) (end t6 h31) (end t62 h31) (end t62 h39) (end t66 h39) (end t66 h6) (end t83 h20) (end t83 h63) (end t84 h77) (end t84 h78) (end t93 h6) (end t93 h63) (end t98 h51) (end t98 h66) (end t99 h6) (end t99 h66) (free t116) (free t133) (free t15) (free t172) (free t186) (free t187) (free t197) (free t198) (free t199) (free t22) (free t31) (free t36) (free t6) (free t62) (free t66) (free t83) (free t84) (free t93) (free t98) (free t99) (link h20 h25) (link h20 h66) (link h25 h20) (link h25 h6) (link h31 h63) (link h31 h78) (link h39 h51) (link h39 h77) (link h51 h39) (link h51 h78) (link h6 h25) (link h6 h63) (link h63 h31) (link h63 h6) (link h66 h20) (link h66 h77) (link h77 h39) (link h77 h66) (link h78 h31) (link h78 h51) (open x3) (ready a10) (ready a24) (ready a4) (ready a7))
  (:goal (and (done x3))))
