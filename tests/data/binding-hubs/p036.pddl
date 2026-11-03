; agents = 3, jobs = 3, hubs = 8
(define (problem p036) (:domain binding-hubs)
  (:objects a10 a18 a22 - agent
            t18 t19 t26 t32 t34 t43 t57 t61 t95 t102 t108 t134 t138 t140 t145 t146 - tool
            h1 h4 h32 h38 h39 h42 h59 h74 - hub
            x3 x6 x15 - job)
  (:init (end t102 h59) (end t102 h74) (end t108 h32) (end t108 h59) (end t134 h39) (end t134 h59) (end t138 h38) (end t138 h42) (end t140 h38) (end t140 h74) (end t145 h1) (end t145 h39) (end t146 h39) (end t146 h74) (end t18 h32) (end t18 h39) (end t19 h1) (end t19 h42) (end t26 h32) (end t26 h74) (end t32 h4) (end t32 h59) (end t34 h1) (end t34 h4) (end t43 h38) (end t43 h4) (end t57 h32) (end t57 h42) (end t61 h1) (end t61 h38) (end t95 h4) (end t95 h42) (free t102) (free t108) (free t134) (free t138) (free t140) (free t145) (free t146) (free t18) (free t19) (free t26) (free t32) (free t34) (free t43) (free t57) (free t61) (free t95) (link h1 h38) (link h1 h59) (link h32 h38) (link h32 h39) (link h38 h1) (link h38 h32) (link h39 h32) (link h39 h4) (link h4 h39) (link h4 h42) (link h42 h4) (link h42 h74) (link h59 h1) (link h59 h74) (link h74 h42) (link h74 h59) (open x15) (open x3) (open x6) (ready a10) (ready a18) (ready a22))
  (:goal (and (done x15) (done x3) (done x6))))
