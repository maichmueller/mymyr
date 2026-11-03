; agents = 2, jobs = 1, hubs = 10
(define (problem p003) (:domain binding-hubs)
  (:objects a3 a20 - agent
            t10 t13 t18 t27 t28 t33 t45 t59 t62 t66 t76 t101 t108 t126 t132 t134 t153 t181 t185 t190 - tool
            h3 h10 h27 h28 h36 h62 h72 h79 h84 h96 - hub
            x2 - job)
  (:init (end t10 h3) (end t10 h96) (end t101 h36) (end t101 h96) (end t108 h62) (end t108 h72) (end t126 h27) (end t126 h62) (end t13 h79) (end t13 h84) (end t132 h62) (end t132 h79) (end t134 h3) (end t134 h84) (end t153 h28) (end t153 h3) (end t18 h27) (end t18 h84) (end t181 h10) (end t181 h27) (end t185 h28) (end t185 h72) (end t190 h28) (end t190 h79) (end t27 h10) (end t27 h36) (end t28 h27) (end t28 h28) (end t33 h79) (end t33 h96) (end t45 h36) (end t45 h62) (end t59 h10) (end t59 h3) (end t62 h36) (end t62 h84) (end t66 h10) (end t66 h72) (end t76 h72) (end t76 h96) (free t10) (free t101) (free t108) (free t126) (free t13) (free t132) (free t134) (free t153) (free t18) (free t181) (free t185) (free t190) (free t27) (free t28) (free t33) (free t45) (free t59) (free t62) (free t66) (free t76) (link h10 h36) (link h10 h79) (link h27 h28) (link h27 h96) (link h28 h27) (link h28 h36) (link h3 h62) (link h3 h96) (link h36 h10) (link h36 h28) (link h62 h3) (link h62 h72) (link h72 h62) (link h72 h84) (link h79 h10) (link h79 h84) (link h84 h72) (link h84 h79) (link h96 h27) (link h96 h3) (open x2) (ready a20) (ready a3))
  (:goal (and (done x2))))
