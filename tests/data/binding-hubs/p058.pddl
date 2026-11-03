; agents = 4, jobs = 3, hubs = 8
(define (problem p058) (:domain binding-hubs)
  (:objects a17 a22 a23 a34 - agent
            t4 t15 t19 t28 t51 t54 t62 t72 t110 t113 t117 t126 t128 t138 t141 t143 - tool
            h5 h9 h13 h20 h23 h43 h45 h47 - hub
            x8 x27 x28 - job)
  (:init (end t110 h23) (end t110 h9) (end t113 h13) (end t113 h9) (end t117 h20) (end t117 h9) (end t126 h43) (end t126 h47) (end t128 h20) (end t128 h43) (end t138 h13) (end t138 h43) (end t141 h23) (end t141 h45) (end t143 h45) (end t143 h47) (end t15 h47) (end t15 h5) (end t19 h43) (end t19 h9) (end t28 h20) (end t28 h5) (end t4 h45) (end t4 h5) (end t51 h13) (end t51 h20) (end t54 h23) (end t54 h47) (end t62 h13) (end t62 h45) (end t72 h23) (end t72 h5) (free t110) (free t113) (free t117) (free t126) (free t128) (free t138) (free t141) (free t143) (free t15) (free t19) (free t28) (free t4) (free t51) (free t54) (free t62) (free t72) (link h13 h23) (link h13 h43) (link h20 h47) (link h20 h9) (link h23 h13) (link h23 h47) (link h43 h13) (link h43 h5) (link h45 h5) (link h45 h9) (link h47 h20) (link h47 h23) (link h5 h43) (link h5 h45) (link h9 h20) (link h9 h45) (open x27) (open x28) (open x8) (ready a17) (ready a22) (ready a23) (ready a34))
  (:goal (and (done x27) (done x28) (done x8))))
