; agents = 3, jobs = 3, hubs = 10
(define (problem p033) (:domain binding-hubs)
  (:objects a1 a19 a21 - agent
            t22 t24 t32 t35 t51 t62 t63 t69 t85 t86 t93 t107 t116 t119 t123 t148 t169 t179 t187 t199 - tool
            h9 h19 h22 h34 h43 h70 h72 h73 h77 h96 - hub
            x4 x27 x30 - job)
  (:init (end t107 h9) (end t107 h96) (end t116 h43) (end t116 h72) (end t119 h22) (end t119 h73) (end t123 h34) (end t123 h70) (end t148 h22) (end t148 h72) (end t169 h70) (end t169 h73) (end t179 h34) (end t179 h43) (end t187 h77) (end t187 h9) (end t199 h22) (end t199 h34) (end t22 h43) (end t22 h96) (end t24 h43) (end t24 h73) (end t32 h70) (end t32 h96) (end t35 h19) (end t35 h77) (end t51 h72) (end t51 h9) (end t62 h19) (end t62 h34) (end t63 h73) (end t63 h9) (end t69 h19) (end t69 h72) (end t85 h19) (end t85 h96) (end t86 h22) (end t86 h77) (end t93 h70) (end t93 h77) (free t107) (free t116) (free t119) (free t123) (free t148) (free t169) (free t179) (free t187) (free t199) (free t22) (free t24) (free t32) (free t35) (free t51) (free t62) (free t63) (free t69) (free t85) (free t86) (free t93) (link h19 h73) (link h19 h77) (link h22 h34) (link h22 h96) (link h34 h22) (link h34 h9) (link h43 h77) (link h43 h96) (link h70 h72) (link h70 h73) (link h72 h70) (link h72 h9) (link h73 h19) (link h73 h70) (link h77 h19) (link h77 h43) (link h9 h34) (link h9 h72) (link h96 h22) (link h96 h43) (open x27) (open x30) (open x4) (ready a1) (ready a19) (ready a21))
  (:goal (and (done x27) (done x30) (done x4))))
