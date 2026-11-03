; agents = 2, jobs = 2, hubs = 8
(define (problem p011) (:domain binding-hubs)
  (:objects a6 a18 - agent
            t37 t52 t54 t66 t70 t78 t80 t98 t128 t132 t134 t135 t140 t144 t147 t157 - tool
            h1 h4 h15 h49 h53 h65 h69 h79 - hub
            x2 x17 - job)
  (:init (end t128 h15) (end t128 h4) (end t132 h15) (end t132 h65) (end t134 h53) (end t134 h69) (end t135 h15) (end t135 h79) (end t140 h4) (end t140 h69) (end t144 h1) (end t144 h49) (end t147 h49) (end t147 h69) (end t157 h15) (end t157 h49) (end t37 h49) (end t37 h53) (end t52 h65) (end t52 h79) (end t54 h4) (end t54 h65) (end t66 h4) (end t66 h79) (end t70 h1) (end t70 h53) (end t78 h1) (end t78 h79) (end t80 h53) (end t80 h65) (end t98 h1) (end t98 h69) (free t128) (free t132) (free t134) (free t135) (free t140) (free t144) (free t147) (free t157) (free t37) (free t52) (free t54) (free t66) (free t70) (free t78) (free t80) (free t98) (link h1 h49) (link h1 h65) (link h15 h65) (link h15 h69) (link h4 h49) (link h4 h79) (link h49 h1) (link h49 h4) (link h53 h69) (link h53 h79) (link h65 h1) (link h65 h15) (link h69 h15) (link h69 h53) (link h79 h4) (link h79 h53) (open x17) (open x2) (ready a18) (ready a6))
  (:goal (and (done x17) (done x2))))
