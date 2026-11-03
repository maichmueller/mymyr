; agents = 4, jobs = 4, hubs = 8
(define (problem p067) (:domain binding-hubs)
  (:objects a8 a23 a26 a36 - agent
            t9 t25 t30 t32 t36 t40 t55 t71 t80 t83 t105 t107 t111 t121 t138 t144 - tool
            h26 h49 h50 h56 h62 h64 h66 h78 - hub
            x7 x11 x36 x38 - job)
  (:init (end t105 h26) (end t105 h50) (end t107 h26) (end t107 h56) (end t111 h62) (end t111 h78) (end t121 h49) (end t121 h56) (end t138 h66) (end t138 h78) (end t144 h64) (end t144 h78) (end t25 h49) (end t25 h78) (end t30 h50) (end t30 h56) (end t32 h49) (end t32 h50) (end t36 h56) (end t36 h62) (end t40 h26) (end t40 h66) (end t55 h62) (end t55 h66) (end t71 h62) (end t71 h64) (end t80 h26) (end t80 h49) (end t83 h50) (end t83 h64) (end t9 h64) (end t9 h66) (free t105) (free t107) (free t111) (free t121) (free t138) (free t144) (free t25) (free t30) (free t32) (free t36) (free t40) (free t55) (free t71) (free t80) (free t83) (free t9) (link h26 h49) (link h26 h64) (link h49 h26) (link h49 h62) (link h50 h56) (link h50 h66) (link h56 h50) (link h56 h78) (link h62 h49) (link h62 h66) (link h64 h26) (link h64 h78) (link h66 h50) (link h66 h62) (link h78 h56) (link h78 h64) (open x11) (open x36) (open x38) (open x7) (ready a23) (ready a26) (ready a36) (ready a8))
  (:goal (and (done x11) (done x36) (done x38) (done x7))))
