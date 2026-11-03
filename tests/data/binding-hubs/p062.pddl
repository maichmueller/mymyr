; agents = 4, jobs = 3, hubs = 8
(define (problem p062) (:domain binding-hubs)
  (:objects a14 a26 a30 a34 - agent
            t1 t22 t25 t31 t37 t43 t54 t65 t95 t102 t111 t122 t128 t132 t150 t160 - tool
            h3 h4 h14 h20 h45 h47 h48 h60 - hub
            x4 x14 x18 - job)
  (:init (end t1 h4) (end t1 h60) (end t102 h3) (end t102 h4) (end t111 h45) (end t111 h48) (end t122 h20) (end t122 h4) (end t128 h4) (end t128 h47) (end t132 h20) (end t132 h47) (end t150 h20) (end t150 h60) (end t160 h3) (end t160 h45) (end t22 h45) (end t22 h47) (end t25 h14) (end t25 h48) (end t31 h14) (end t31 h3) (end t37 h48) (end t37 h60) (end t43 h14) (end t43 h20) (end t54 h3) (end t54 h48) (end t65 h47) (end t65 h60) (end t95 h14) (end t95 h45) (free t1) (free t102) (free t111) (free t122) (free t128) (free t132) (free t150) (free t160) (free t22) (free t25) (free t31) (free t37) (free t43) (free t54) (free t65) (free t95) (link h14 h4) (link h14 h45) (link h20 h3) (link h20 h60) (link h3 h20) (link h3 h48) (link h4 h14) (link h4 h47) (link h45 h14) (link h45 h60) (link h47 h4) (link h47 h48) (link h48 h3) (link h48 h47) (link h60 h20) (link h60 h45) (open x14) (open x18) (open x4) (ready a14) (ready a26) (ready a30) (ready a34))
  (:goal (and (done x14) (done x18) (done x4))))
