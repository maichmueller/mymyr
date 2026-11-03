; agents = 4, jobs = 1, hubs = 8
(define (problem p042) (:domain binding-hubs)
  (:objects a8 a17 a19 a36 - agent
            t11 t15 t23 t42 t56 t59 t62 t80 t87 t88 t94 t104 t108 t119 t150 t157 - tool
            h23 h28 h49 h51 h60 h73 h74 h76 - hub
            x1 - job)
  (:init (end t104 h51) (end t104 h73) (end t108 h23) (end t108 h60) (end t11 h49) (end t11 h76) (end t119 h28) (end t119 h51) (end t15 h23) (end t15 h49) (end t150 h28) (end t150 h76) (end t157 h51) (end t157 h74) (end t23 h49) (end t23 h74) (end t42 h23) (end t42 h74) (end t56 h28) (end t56 h73) (end t59 h60) (end t59 h73) (end t62 h73) (end t62 h76) (end t80 h23) (end t80 h28) (end t87 h49) (end t87 h60) (end t88 h60) (end t88 h74) (end t94 h51) (end t94 h76) (free t104) (free t108) (free t11) (free t119) (free t15) (free t150) (free t157) (free t23) (free t42) (free t56) (free t59) (free t62) (free t80) (free t87) (free t88) (free t94) (link h23 h60) (link h23 h76) (link h28 h49) (link h28 h51) (link h49 h28) (link h49 h74) (link h51 h28) (link h51 h60) (link h60 h23) (link h60 h51) (link h73 h74) (link h73 h76) (link h74 h49) (link h74 h73) (link h76 h23) (link h76 h73) (open x1) (ready a17) (ready a19) (ready a36) (ready a8))
  (:goal (and (done x1))))
