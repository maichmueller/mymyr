; agents = 4, jobs = 3, hubs = 10
(define (problem p063) (:domain binding-hubs)
  (:objects a19 a20 a23 a33 - agent
            t14 t27 t36 t37 t40 t44 t56 t63 t64 t106 t116 t118 t120 t129 t135 t143 t145 t151 t152 t200 - tool
            h2 h15 h16 h20 h23 h30 h53 h79 h80 h92 - hub
            x13 x23 x24 - job)
  (:init (end t106 h15) (end t106 h20) (end t116 h30) (end t116 h92) (end t118 h20) (end t118 h53) (end t120 h23) (end t120 h79) (end t129 h16) (end t129 h23) (end t135 h15) (end t135 h92) (end t14 h53) (end t14 h92) (end t143 h15) (end t143 h2) (end t145 h20) (end t145 h79) (end t151 h2) (end t151 h79) (end t152 h2) (end t152 h30) (end t200 h23) (end t200 h53) (end t27 h23) (end t27 h30) (end t36 h15) (end t36 h80) (end t37 h30) (end t37 h80) (end t40 h2) (end t40 h53) (end t44 h16) (end t44 h20) (end t56 h16) (end t56 h80) (end t63 h79) (end t63 h80) (end t64 h16) (end t64 h92) (free t106) (free t116) (free t118) (free t120) (free t129) (free t135) (free t14) (free t143) (free t145) (free t151) (free t152) (free t200) (free t27) (free t36) (free t37) (free t40) (free t44) (free t56) (free t63) (free t64) (link h15 h20) (link h15 h23) (link h16 h2) (link h16 h92) (link h2 h16) (link h2 h30) (link h20 h15) (link h20 h30) (link h23 h15) (link h23 h53) (link h30 h2) (link h30 h20) (link h53 h23) (link h53 h80) (link h79 h80) (link h79 h92) (link h80 h53) (link h80 h79) (link h92 h16) (link h92 h79) (open x13) (open x23) (open x24) (ready a19) (ready a20) (ready a23) (ready a33))
  (:goal (and (done x13) (done x23) (done x24))))
