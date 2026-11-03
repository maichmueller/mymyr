; agents = 4, jobs = 4, hubs = 8
(define (problem p070) (:domain binding-hubs)
  (:objects a11 a17 a35 a38 - agent
            t14 t20 t56 t60 t70 t79 t89 t90 t93 t95 t110 t116 t124 t130 t139 t145 - tool
            h1 h10 h14 h15 h39 h71 h73 h74 - hub
            x8 x13 x20 x24 - job)
  (:init (end t110 h1) (end t110 h39) (end t116 h10) (end t116 h14) (end t124 h1) (end t124 h74) (end t130 h73) (end t130 h74) (end t139 h10) (end t139 h71) (end t14 h10) (end t14 h74) (end t145 h1) (end t145 h15) (end t20 h10) (end t20 h73) (end t56 h14) (end t56 h15) (end t60 h15) (end t60 h39) (end t70 h39) (end t70 h71) (end t79 h15) (end t79 h71) (end t89 h1) (end t89 h71) (end t90 h14) (end t90 h74) (end t93 h14) (end t93 h73) (end t95 h39) (end t95 h73) (free t110) (free t116) (free t124) (free t130) (free t139) (free t14) (free t145) (free t20) (free t56) (free t60) (free t70) (free t79) (free t89) (free t90) (free t93) (free t95) (link h1 h14) (link h1 h39) (link h10 h39) (link h10 h74) (link h14 h1) (link h14 h73) (link h15 h71) (link h15 h74) (link h39 h1) (link h39 h10) (link h71 h15) (link h71 h73) (link h73 h14) (link h73 h71) (link h74 h10) (link h74 h15) (open x13) (open x20) (open x24) (open x8) (ready a11) (ready a17) (ready a35) (ready a38))
  (:goal (and (done x13) (done x20) (done x24) (done x8))))
