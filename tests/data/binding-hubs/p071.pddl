; agents = 4, jobs = 4, hubs = 8
(define (problem p071) (:domain binding-hubs)
  (:objects a6 a23 a34 a40 - agent
            t8 t21 t23 t27 t35 t36 t50 t51 t66 t77 t86 t107 t125 t128 t143 t144 - tool
            h3 h20 h29 h35 h55 h63 h70 h78 - hub
            x21 x37 x38 x40 - job)
  (:init (end t107 h3) (end t107 h63) (end t125 h35) (end t125 h78) (end t128 h29) (end t128 h35) (end t143 h29) (end t143 h55) (end t144 h3) (end t144 h35) (end t21 h3) (end t21 h70) (end t23 h20) (end t23 h63) (end t27 h55) (end t27 h78) (end t35 h29) (end t35 h78) (end t36 h29) (end t36 h70) (end t50 h20) (end t50 h55) (end t51 h20) (end t51 h70) (end t66 h20) (end t66 h3) (end t77 h35) (end t77 h55) (end t8 h63) (end t8 h78) (end t86 h63) (end t86 h70) (free t107) (free t125) (free t128) (free t143) (free t144) (free t21) (free t23) (free t27) (free t35) (free t36) (free t50) (free t51) (free t66) (free t77) (free t8) (free t86) (link h20 h35) (link h20 h70) (link h29 h35) (link h29 h63) (link h3 h55) (link h3 h63) (link h35 h20) (link h35 h29) (link h55 h3) (link h55 h78) (link h63 h29) (link h63 h3) (link h70 h20) (link h70 h78) (link h78 h55) (link h78 h70) (open x21) (open x37) (open x38) (open x40) (ready a23) (ready a34) (ready a40) (ready a6))
  (:goal (and (done x21) (done x37) (done x38) (done x40))))
