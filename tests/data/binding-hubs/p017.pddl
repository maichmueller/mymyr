; agents = 3, jobs = 1, hubs = 8
(define (problem p017) (:domain binding-hubs)
  (:objects a13 a16 a17 - agent
            t6 t13 t28 t30 t31 t51 t55 t61 t66 t97 t102 t103 t114 t130 t148 t158 - tool
            h2 h13 h33 h34 h36 h44 h70 h71 - hub
            x9 - job)
  (:init (end t102 h36) (end t102 h44) (end t103 h34) (end t103 h70) (end t114 h13) (end t114 h71) (end t13 h13) (end t13 h44) (end t130 h2) (end t130 h36) (end t148 h70) (end t148 h71) (end t158 h34) (end t158 h44) (end t28 h33) (end t28 h71) (end t30 h13) (end t30 h36) (end t31 h33) (end t31 h34) (end t51 h34) (end t51 h71) (end t55 h13) (end t55 h2) (end t6 h2) (end t6 h33) (end t61 h36) (end t61 h70) (end t66 h2) (end t66 h44) (end t97 h33) (end t97 h70) (free t102) (free t103) (free t114) (free t13) (free t130) (free t148) (free t158) (free t28) (free t30) (free t31) (free t51) (free t55) (free t6) (free t61) (free t66) (free t97) (link h13 h2) (link h13 h34) (link h2 h13) (link h2 h70) (link h33 h34) (link h33 h36) (link h34 h13) (link h34 h33) (link h36 h33) (link h36 h44) (link h44 h36) (link h44 h71) (link h70 h2) (link h70 h71) (link h71 h44) (link h71 h70) (open x9) (ready a13) (ready a16) (ready a17))
  (:goal (and (done x9))))
