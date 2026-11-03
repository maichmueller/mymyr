; agents = 3, jobs = 2, hubs = 10
(define (problem p026) (:domain binding-hubs)
  (:objects a13 a26 a27 - agent
            t3 t7 t23 t29 t38 t42 t43 t63 t83 t93 t97 t108 t119 t135 t139 t163 t164 t176 t182 t190 - tool
            h8 h27 h36 h38 h40 h50 h55 h61 h68 h94 - hub
            x4 x10 - job)
  (:init (end t108 h36) (end t108 h40) (end t119 h55) (end t119 h68) (end t135 h27) (end t135 h68) (end t139 h36) (end t139 h38) (end t163 h27) (end t163 h36) (end t164 h61) (end t164 h8) (end t176 h55) (end t176 h61) (end t182 h36) (end t182 h55) (end t190 h40) (end t190 h50) (end t23 h27) (end t23 h50) (end t29 h38) (end t29 h61) (end t3 h68) (end t3 h8) (end t38 h8) (end t38 h94) (end t42 h55) (end t42 h94) (end t43 h40) (end t43 h68) (end t63 h27) (end t63 h61) (end t7 h40) (end t7 h94) (end t83 h38) (end t83 h94) (end t93 h38) (end t93 h50) (end t97 h50) (end t97 h8) (free t108) (free t119) (free t135) (free t139) (free t163) (free t164) (free t176) (free t182) (free t190) (free t23) (free t29) (free t3) (free t38) (free t42) (free t43) (free t63) (free t7) (free t83) (free t93) (free t97) (link h27 h50) (link h27 h94) (link h36 h38) (link h36 h8) (link h38 h36) (link h38 h68) (link h40 h61) (link h40 h94) (link h50 h27) (link h50 h55) (link h55 h50) (link h55 h68) (link h61 h40) (link h61 h8) (link h68 h38) (link h68 h55) (link h8 h36) (link h8 h61) (link h94 h27) (link h94 h40) (open x10) (open x4) (ready a13) (ready a26) (ready a27))
  (:goal (and (done x10) (done x4))))
