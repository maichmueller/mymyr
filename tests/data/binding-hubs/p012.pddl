; agents = 2, jobs = 2, hubs = 8
(define (problem p012) (:domain binding-hubs)
  (:objects a15 a18 - agent
            t8 t28 t33 t50 t64 t69 t82 t87 t104 t106 t108 t112 t122 t127 t135 t159 - tool
            h2 h3 h5 h17 h20 h25 h30 h34 - hub
            x10 x11 - job)
  (:init (end t104 h3) (end t104 h5) (end t106 h17) (end t106 h34) (end t108 h17) (end t108 h20) (end t112 h2) (end t112 h25) (end t122 h34) (end t122 h5) (end t127 h17) (end t127 h25) (end t135 h20) (end t135 h34) (end t159 h3) (end t159 h30) (end t28 h2) (end t28 h3) (end t33 h20) (end t33 h3) (end t50 h25) (end t50 h34) (end t64 h2) (end t64 h30) (end t69 h20) (end t69 h25) (end t8 h2) (end t8 h5) (end t82 h17) (end t82 h30) (end t87 h30) (end t87 h5) (free t104) (free t106) (free t108) (free t112) (free t122) (free t127) (free t135) (free t159) (free t28) (free t33) (free t50) (free t64) (free t69) (free t8) (free t82) (free t87) (link h17 h3) (link h17 h34) (link h2 h30) (link h2 h34) (link h20 h25) (link h20 h30) (link h25 h20) (link h25 h5) (link h3 h17) (link h3 h5) (link h30 h2) (link h30 h20) (link h34 h17) (link h34 h2) (link h5 h25) (link h5 h3) (open x10) (open x11) (ready a15) (ready a18))
  (:goal (and (done x10) (done x11))))
