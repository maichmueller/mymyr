; agents = 4, jobs = 2, hubs = 8
(define (problem p049) (:domain binding-hubs)
  (:objects a15 a16 a17 a34 - agent
            t1 t5 t21 t27 t54 t62 t68 t81 t86 t89 t101 t127 t140 t141 t144 t160 - tool
            h21 h25 h43 h51 h63 h73 h77 h79 - hub
            x10 x12 - job)
  (:init (end t1 h21) (end t1 h51) (end t101 h43) (end t101 h77) (end t127 h21) (end t127 h77) (end t140 h25) (end t140 h79) (end t141 h51) (end t141 h79) (end t144 h73) (end t144 h77) (end t160 h63) (end t160 h79) (end t21 h25) (end t21 h43) (end t27 h51) (end t27 h77) (end t5 h21) (end t5 h43) (end t54 h73) (end t54 h79) (end t62 h63) (end t62 h73) (end t68 h21) (end t68 h63) (end t81 h25) (end t81 h63) (end t86 h43) (end t86 h51) (end t89 h25) (end t89 h73) (free t1) (free t101) (free t127) (free t140) (free t141) (free t144) (free t160) (free t21) (free t27) (free t5) (free t54) (free t62) (free t68) (free t81) (free t86) (free t89) (link h21 h25) (link h21 h51) (link h25 h21) (link h25 h79) (link h43 h63) (link h43 h77) (link h51 h21) (link h51 h73) (link h63 h43) (link h63 h73) (link h73 h51) (link h73 h63) (link h77 h43) (link h77 h79) (link h79 h25) (link h79 h77) (open x10) (open x12) (ready a15) (ready a16) (ready a17) (ready a34))
  (:goal (and (done x10) (done x12))))
