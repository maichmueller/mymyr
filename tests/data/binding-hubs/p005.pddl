; agents = 2, jobs = 1, hubs = 8
(define (problem p005) (:domain binding-hubs)
  (:objects a3 a20 - agent
            t4 t8 t12 t20 t22 t34 t88 t91 t95 t107 t116 t122 t128 t140 t147 t160 - tool
            h2 h10 h11 h12 h15 h49 h75 h80 - hub
            x5 - job)
  (:init (end t107 h12) (end t107 h80) (end t116 h49) (end t116 h80) (end t12 h12) (end t12 h15) (end t122 h10) (end t122 h15) (end t128 h10) (end t128 h49) (end t140 h2) (end t140 h75) (end t147 h11) (end t147 h80) (end t160 h15) (end t160 h2) (end t20 h11) (end t20 h49) (end t22 h12) (end t22 h2) (end t34 h2) (end t34 h49) (end t4 h12) (end t4 h75) (end t8 h15) (end t8 h75) (end t88 h10) (end t88 h80) (end t91 h11) (end t91 h75) (end t95 h10) (end t95 h11) (free t107) (free t116) (free t12) (free t122) (free t128) (free t140) (free t147) (free t160) (free t20) (free t22) (free t34) (free t4) (free t8) (free t88) (free t91) (free t95) (link h10 h11) (link h10 h2) (link h11 h10) (link h11 h12) (link h12 h11) (link h12 h15) (link h15 h12) (link h15 h49) (link h2 h10) (link h2 h75) (link h49 h15) (link h49 h80) (link h75 h2) (link h75 h80) (link h80 h49) (link h80 h75) (open x5) (ready a20) (ready a3))
  (:goal (and (done x5))))
