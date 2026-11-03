; agents = 4, jobs = 2, hubs = 10
(define (problem p054) (:domain binding-hubs)
  (:objects a2 a18 a23 a36 - agent
            t3 t17 t26 t54 t63 t68 t79 t80 t94 t103 t120 t124 t127 t132 t135 t137 t171 t181 t189 t197 - tool
            h2 h4 h19 h30 h33 h42 h50 h53 h74 h80 - hub
            x12 x17 - job)
  (:init (end t103 h33) (end t103 h42) (end t120 h19) (end t120 h30) (end t124 h4) (end t124 h42) (end t127 h30) (end t127 h53) (end t132 h2) (end t132 h74) (end t135 h19) (end t135 h4) (end t137 h53) (end t137 h74) (end t17 h4) (end t17 h80) (end t171 h19) (end t171 h33) (end t181 h2) (end t181 h50) (end t189 h19) (end t189 h50) (end t197 h74) (end t197 h80) (end t26 h30) (end t26 h80) (end t3 h42) (end t3 h50) (end t54 h4) (end t54 h53) (end t63 h42) (end t63 h74) (end t68 h2) (end t68 h30) (end t79 h50) (end t79 h80) (end t80 h33) (end t80 h53) (end t94 h2) (end t94 h33) (free t103) (free t120) (free t124) (free t127) (free t132) (free t135) (free t137) (free t17) (free t171) (free t181) (free t189) (free t197) (free t26) (free t3) (free t54) (free t63) (free t68) (free t79) (free t80) (free t94) (link h19 h30) (link h19 h74) (link h2 h4) (link h2 h74) (link h30 h19) (link h30 h42) (link h33 h53) (link h33 h80) (link h4 h2) (link h4 h80) (link h42 h30) (link h42 h50) (link h50 h42) (link h50 h53) (link h53 h33) (link h53 h50) (link h74 h19) (link h74 h2) (link h80 h33) (link h80 h4) (open x12) (open x17) (ready a18) (ready a2) (ready a23) (ready a36))
  (:goal (and (done x12) (done x17))))
