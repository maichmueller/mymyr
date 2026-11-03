; agents = 4, jobs = 3, hubs = 8
(define (problem p059) (:domain binding-hubs)
  (:objects a6 a14 a21 a35 - agent
            t4 t11 t33 t44 t47 t51 t54 t72 t81 t87 t97 t111 t120 t144 t149 t155 - tool
            h15 h17 h26 h34 h36 h40 h45 h46 - hub
            x1 x3 x23 - job)
  (:init (end t11 h26) (end t11 h34) (end t111 h45) (end t111 h46) (end t120 h15) (end t120 h40) (end t144 h15) (end t144 h34) (end t149 h36) (end t149 h46) (end t155 h26) (end t155 h45) (end t33 h15) (end t33 h26) (end t4 h17) (end t4 h34) (end t44 h17) (end t44 h45) (end t47 h36) (end t47 h40) (end t51 h17) (end t51 h36) (end t54 h17) (end t54 h46) (end t72 h34) (end t72 h40) (end t81 h15) (end t81 h46) (end t87 h36) (end t87 h45) (end t97 h26) (end t97 h40) (free t11) (free t111) (free t120) (free t144) (free t149) (free t155) (free t33) (free t4) (free t44) (free t47) (free t51) (free t54) (free t72) (free t81) (free t87) (free t97) (link h15 h34) (link h15 h36) (link h17 h26) (link h17 h36) (link h26 h17) (link h26 h40) (link h34 h15) (link h34 h45) (link h36 h15) (link h36 h17) (link h40 h26) (link h40 h46) (link h45 h34) (link h45 h46) (link h46 h40) (link h46 h45) (open x1) (open x23) (open x3) (ready a14) (ready a21) (ready a35) (ready a6))
  (:goal (and (done x1) (done x23) (done x3))))
