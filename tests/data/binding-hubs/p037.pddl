; agents = 3, jobs = 3, hubs = 10
(define (problem p037) (:domain binding-hubs)
  (:objects a10 a11 a29 - agent
            t5 t16 t37 t45 t51 t52 t53 t76 t81 t84 t91 t121 t128 t137 t144 t162 t163 t178 t179 t199 - tool
            h7 h14 h25 h31 h36 h40 h41 h54 h84 h93 - hub
            x5 x23 x29 - job)
  (:init (end t121 h36) (end t121 h54) (end t128 h7) (end t128 h84) (end t137 h36) (end t137 h84) (end t144 h41) (end t144 h7) (end t16 h14) (end t16 h41) (end t162 h14) (end t162 h54) (end t163 h31) (end t163 h40) (end t178 h84) (end t178 h93) (end t179 h25) (end t179 h93) (end t199 h41) (end t199 h93) (end t37 h14) (end t37 h40) (end t45 h54) (end t45 h7) (end t5 h25) (end t5 h36) (end t51 h36) (end t51 h40) (end t52 h14) (end t52 h25) (end t53 h31) (end t53 h84) (end t76 h40) (end t76 h7) (end t81 h54) (end t81 h93) (end t84 h25) (end t84 h31) (end t91 h31) (end t91 h41) (free t121) (free t128) (free t137) (free t144) (free t16) (free t162) (free t163) (free t178) (free t179) (free t199) (free t37) (free t45) (free t5) (free t51) (free t52) (free t53) (free t76) (free t81) (free t84) (free t91) (link h14 h40) (link h14 h84) (link h25 h31) (link h25 h7) (link h31 h25) (link h31 h54) (link h36 h41) (link h36 h54) (link h40 h14) (link h40 h93) (link h41 h36) (link h41 h93) (link h54 h31) (link h54 h36) (link h7 h25) (link h7 h84) (link h84 h14) (link h84 h7) (link h93 h40) (link h93 h41) (open x23) (open x29) (open x5) (ready a10) (ready a11) (ready a29))
  (:goal (and (done x23) (done x29) (done x5))))
