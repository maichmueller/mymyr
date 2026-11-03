; agents = 4, jobs = 1, hubs = 10
(define (problem p046) (:domain binding-hubs)
  (:objects a8 a14 a23 a36 - agent
            t16 t24 t31 t41 t57 t61 t67 t72 t73 t85 t96 t99 t101 t110 t126 t127 t128 t162 t181 t199 - tool
            h22 h26 h31 h45 h52 h61 h67 h86 h92 h99 - hub
            x2 - job)
  (:init (end t101 h86) (end t101 h99) (end t110 h22) (end t110 h26) (end t126 h67) (end t126 h86) (end t127 h22) (end t127 h92) (end t128 h45) (end t128 h52) (end t16 h31) (end t16 h45) (end t162 h86) (end t162 h92) (end t181 h26) (end t181 h86) (end t199 h26) (end t199 h52) (end t24 h52) (end t24 h99) (end t31 h45) (end t31 h61) (end t41 h22) (end t41 h67) (end t57 h31) (end t57 h99) (end t61 h22) (end t61 h45) (end t67 h31) (end t67 h67) (end t72 h61) (end t72 h92) (end t73 h61) (end t73 h67) (end t85 h26) (end t85 h31) (end t96 h61) (end t96 h99) (end t99 h52) (end t99 h92) (free t101) (free t110) (free t126) (free t127) (free t128) (free t16) (free t162) (free t181) (free t199) (free t24) (free t31) (free t41) (free t57) (free t61) (free t67) (free t72) (free t73) (free t85) (free t96) (free t99) (link h22 h45) (link h22 h99) (link h26 h52) (link h26 h61) (link h31 h67) (link h31 h92) (link h45 h22) (link h45 h86) (link h52 h26) (link h52 h67) (link h61 h26) (link h61 h99) (link h67 h31) (link h67 h52) (link h86 h45) (link h86 h92) (link h92 h31) (link h92 h86) (link h99 h22) (link h99 h61) (open x2) (ready a14) (ready a23) (ready a36) (ready a8))
  (:goal (and (done x2))))
