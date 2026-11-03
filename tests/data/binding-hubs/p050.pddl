; agents = 4, jobs = 2, hubs = 8
(define (problem p050) (:domain binding-hubs)
  (:objects a12 a16 a21 a24 - agent
            t11 t14 t17 t18 t27 t29 t33 t42 t65 t80 t89 t106 t116 t124 t132 t160 - tool
            h24 h33 h42 h48 h51 h65 h77 h80 - hub
            x7 x11 - job)
  (:init (end t106 h51) (end t106 h77) (end t11 h48) (end t11 h51) (end t116 h33) (end t116 h42) (end t124 h24) (end t124 h77) (end t132 h51) (end t132 h65) (end t14 h48) (end t14 h65) (end t160 h24) (end t160 h80) (end t17 h48) (end t17 h77) (end t18 h24) (end t18 h42) (end t27 h33) (end t27 h51) (end t29 h48) (end t29 h80) (end t33 h33) (end t33 h80) (end t42 h42) (end t42 h65) (end t65 h24) (end t65 h33) (end t80 h65) (end t80 h77) (end t89 h42) (end t89 h80) (free t106) (free t11) (free t116) (free t124) (free t132) (free t14) (free t160) (free t17) (free t18) (free t27) (free t29) (free t33) (free t42) (free t65) (free t80) (free t89) (link h24 h42) (link h24 h51) (link h33 h77) (link h33 h80) (link h42 h24) (link h42 h48) (link h48 h42) (link h48 h77) (link h51 h24) (link h51 h65) (link h65 h51) (link h65 h80) (link h77 h33) (link h77 h48) (link h80 h33) (link h80 h65) (open x11) (open x7) (ready a12) (ready a16) (ready a21) (ready a24))
  (:goal (and (done x11) (done x7))))
