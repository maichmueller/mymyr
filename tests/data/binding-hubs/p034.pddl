; agents = 3, jobs = 3, hubs = 10
(define (problem p034) (:domain binding-hubs)
  (:objects a2 a13 a15 - agent
            t5 t9 t36 t37 t41 t47 t52 t79 t98 t106 t120 t136 t152 t158 t164 t169 t172 t189 t194 t196 - tool
            h16 h17 h23 h32 h34 h44 h50 h52 h67 h73 - hub
            x11 x19 x30 - job)
  (:init (end t106 h44) (end t106 h50) (end t120 h16) (end t120 h67) (end t136 h23) (end t136 h32) (end t152 h17) (end t152 h23) (end t158 h52) (end t158 h67) (end t164 h34) (end t164 h44) (end t169 h52) (end t169 h73) (end t172 h50) (end t172 h52) (end t189 h16) (end t189 h23) (end t194 h32) (end t194 h34) (end t196 h32) (end t196 h67) (end t36 h23) (end t36 h44) (end t37 h34) (end t37 h52) (end t41 h16) (end t41 h34) (end t47 h16) (end t47 h50) (end t5 h17) (end t5 h50) (end t52 h32) (end t52 h73) (end t79 h17) (end t79 h67) (end t9 h17) (end t9 h73) (end t98 h44) (end t98 h73) (free t106) (free t120) (free t136) (free t152) (free t158) (free t164) (free t169) (free t172) (free t189) (free t194) (free t196) (free t36) (free t37) (free t41) (free t47) (free t5) (free t52) (free t79) (free t9) (free t98) (link h16 h50) (link h16 h73) (link h17 h34) (link h17 h67) (link h23 h44) (link h23 h52) (link h32 h34) (link h32 h50) (link h34 h17) (link h34 h32) (link h44 h23) (link h44 h67) (link h50 h16) (link h50 h32) (link h52 h23) (link h52 h73) (link h67 h17) (link h67 h44) (link h73 h16) (link h73 h52) (open x11) (open x19) (open x30) (ready a13) (ready a15) (ready a2))
  (:goal (and (done x11) (done x19) (done x30))))
