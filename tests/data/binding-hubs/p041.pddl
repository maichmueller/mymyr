; agents = 4, jobs = 1, hubs = 8
(define (problem p041) (:domain binding-hubs)
  (:objects a11 a30 a32 a39 - agent
            t1 t11 t16 t33 t35 t38 t56 t62 t81 t94 t106 t117 t120 t126 t134 t154 - tool
            h12 h19 h39 h51 h53 h61 h68 h72 - hub
            x7 - job)
  (:init (end t1 h19) (end t1 h39) (end t106 h51) (end t106 h72) (end t11 h19) (end t11 h72) (end t117 h51) (end t117 h53) (end t120 h39) (end t120 h61) (end t126 h51) (end t126 h61) (end t134 h68) (end t134 h72) (end t154 h53) (end t154 h72) (end t16 h53) (end t16 h68) (end t33 h12) (end t33 h39) (end t35 h39) (end t35 h68) (end t38 h19) (end t38 h61) (end t56 h51) (end t56 h68) (end t62 h12) (end t62 h61) (end t81 h12) (end t81 h19) (end t94 h12) (end t94 h53) (free t1) (free t106) (free t11) (free t117) (free t120) (free t126) (free t134) (free t154) (free t16) (free t33) (free t35) (free t38) (free t56) (free t62) (free t81) (free t94) (link h12 h19) (link h12 h51) (link h19 h12) (link h19 h68) (link h39 h61) (link h39 h72) (link h51 h12) (link h51 h72) (link h53 h61) (link h53 h68) (link h61 h39) (link h61 h53) (link h68 h19) (link h68 h53) (link h72 h39) (link h72 h51) (open x7) (ready a11) (ready a30) (ready a32) (ready a39))
  (:goal (and (done x7))))
