; agents = 2, jobs = 1, hubs = 8
(define (problem p007) (:domain binding-hubs)
  (:objects a8 a11 - agent
            t11 t17 t35 t51 t52 t54 t55 t57 t65 t82 t90 t99 t104 t111 t128 t150 - tool
            h15 h22 h28 h36 h53 h58 h61 h80 - hub
            x7 - job)
  (:init (end t104 h22) (end t104 h36) (end t11 h15) (end t11 h28) (end t111 h28) (end t111 h53) (end t128 h36) (end t128 h61) (end t150 h53) (end t150 h61) (end t17 h53) (end t17 h58) (end t35 h15) (end t35 h53) (end t51 h61) (end t51 h80) (end t52 h36) (end t52 h58) (end t54 h22) (end t54 h28) (end t55 h22) (end t55 h61) (end t57 h28) (end t57 h58) (end t65 h15) (end t65 h58) (end t82 h36) (end t82 h80) (end t90 h22) (end t90 h80) (end t99 h15) (end t99 h80) (free t104) (free t11) (free t111) (free t128) (free t150) (free t17) (free t35) (free t51) (free t52) (free t54) (free t55) (free t57) (free t65) (free t82) (free t90) (free t99) (link h15 h22) (link h15 h58) (link h22 h15) (link h22 h36) (link h28 h53) (link h28 h80) (link h36 h22) (link h36 h53) (link h53 h28) (link h53 h36) (link h58 h15) (link h58 h61) (link h61 h58) (link h61 h80) (link h80 h28) (link h80 h61) (open x7) (ready a11) (ready a8))
  (:goal (and (done x7))))
