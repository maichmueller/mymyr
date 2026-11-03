; agents = 2, jobs = 2, hubs = 10
(define (problem p010) (:domain binding-hubs)
  (:objects a7 a8 - agent
            t5 t8 t10 t22 t23 t68 t71 t80 t83 t88 t90 t100 t108 t113 t125 t130 t138 t141 t184 t197 - tool
            h5 h9 h18 h23 h28 h43 h55 h61 h69 h88 - hub
            x5 x17 - job)
  (:init (end t10 h43) (end t10 h61) (end t100 h28) (end t100 h9) (end t108 h28) (end t108 h69) (end t113 h28) (end t113 h88) (end t125 h18) (end t125 h5) (end t130 h23) (end t130 h88) (end t138 h43) (end t138 h88) (end t141 h55) (end t141 h61) (end t184 h23) (end t184 h9) (end t197 h43) (end t197 h9) (end t22 h18) (end t22 h69) (end t23 h5) (end t23 h55) (end t5 h55) (end t5 h69) (end t68 h28) (end t68 h5) (end t71 h55) (end t71 h88) (end t8 h23) (end t8 h61) (end t80 h23) (end t80 h69) (end t83 h18) (end t83 h61) (end t88 h18) (end t88 h9) (end t90 h43) (end t90 h5) (free t10) (free t100) (free t108) (free t113) (free t125) (free t130) (free t138) (free t141) (free t184) (free t197) (free t22) (free t23) (free t5) (free t68) (free t71) (free t8) (free t80) (free t83) (free t88) (free t90) (link h18 h61) (link h18 h88) (link h23 h5) (link h23 h69) (link h28 h61) (link h28 h9) (link h43 h69) (link h43 h88) (link h5 h23) (link h5 h55) (link h55 h5) (link h55 h9) (link h61 h18) (link h61 h28) (link h69 h23) (link h69 h43) (link h88 h18) (link h88 h43) (link h9 h28) (link h9 h55) (open x17) (open x5) (ready a7) (ready a8))
  (:goal (and (done x17) (done x5))))
