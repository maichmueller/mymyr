; agents = 4, jobs = 1, hubs = 10
(define (problem p047) (:domain binding-hubs)
  (:objects a17 a31 a36 a38 - agent
            t6 t7 t14 t17 t19 t31 t65 t70 t77 t79 t84 t94 t97 t101 t104 t121 t127 t150 t180 t185 - tool
            h5 h24 h34 h39 h42 h45 h48 h73 h83 h86 - hub
            x8 - job)
  (:init (end t101 h34) (end t101 h86) (end t104 h39) (end t104 h42) (end t121 h42) (end t121 h83) (end t127 h34) (end t127 h83) (end t14 h73) (end t14 h86) (end t150 h5) (end t150 h86) (end t17 h45) (end t17 h73) (end t180 h42) (end t180 h48) (end t185 h39) (end t185 h5) (end t19 h24) (end t19 h48) (end t31 h24) (end t31 h83) (end t6 h39) (end t6 h73) (end t65 h42) (end t65 h86) (end t7 h73) (end t7 h83) (end t70 h34) (end t70 h48) (end t77 h24) (end t77 h45) (end t79 h45) (end t79 h5) (end t84 h34) (end t84 h45) (end t94 h48) (end t94 h5) (end t97 h24) (end t97 h39) (free t101) (free t104) (free t121) (free t127) (free t14) (free t150) (free t17) (free t180) (free t185) (free t19) (free t31) (free t6) (free t65) (free t7) (free t70) (free t77) (free t79) (free t84) (free t94) (free t97) (link h24 h83) (link h24 h86) (link h34 h39) (link h34 h86) (link h39 h34) (link h39 h73) (link h42 h45) (link h42 h48) (link h45 h42) (link h45 h5) (link h48 h42) (link h48 h73) (link h5 h45) (link h5 h83) (link h73 h39) (link h73 h48) (link h83 h24) (link h83 h5) (link h86 h24) (link h86 h34) (open x8) (ready a17) (ready a31) (ready a36) (ready a38))
  (:goal (and (done x8))))
