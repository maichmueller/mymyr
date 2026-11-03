; agents = 2, jobs = 1, hubs = 10
(define (problem p006) (:domain binding-hubs)
  (:objects a15 a19 - agent
            t6 t8 t22 t23 t29 t59 t80 t113 t119 t124 t128 t132 t133 t139 t154 t158 t169 t179 t193 t200 - tool
            h2 h19 h26 h33 h39 h48 h63 h67 h79 h87 - hub
            x3 - job)
  (:init (end t113 h33) (end t113 h63) (end t119 h2) (end t119 h48) (end t124 h67) (end t124 h87) (end t128 h33) (end t128 h87) (end t132 h33) (end t132 h39) (end t133 h26) (end t133 h67) (end t139 h19) (end t139 h39) (end t154 h39) (end t154 h79) (end t158 h26) (end t158 h48) (end t169 h39) (end t169 h67) (end t179 h2) (end t179 h67) (end t193 h48) (end t193 h63) (end t200 h2) (end t200 h79) (end t22 h79) (end t22 h87) (end t23 h19) (end t23 h26) (end t29 h19) (end t29 h63) (end t59 h26) (end t59 h33) (end t6 h63) (end t6 h79) (end t8 h19) (end t8 h2) (end t80 h48) (end t80 h87) (free t113) (free t119) (free t124) (free t128) (free t132) (free t133) (free t139) (free t154) (free t158) (free t169) (free t179) (free t193) (free t200) (free t22) (free t23) (free t29) (free t59) (free t6) (free t8) (free t80) (link h19 h39) (link h19 h87) (link h2 h33) (link h2 h48) (link h26 h67) (link h26 h79) (link h33 h2) (link h33 h63) (link h39 h19) (link h39 h48) (link h48 h2) (link h48 h39) (link h63 h33) (link h63 h67) (link h67 h26) (link h67 h63) (link h79 h26) (link h79 h87) (link h87 h19) (link h87 h79) (open x3) (ready a15) (ready a19))
  (:goal (and (done x3))))
