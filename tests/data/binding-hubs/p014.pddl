; agents = 2, jobs = 2, hubs = 10
(define (problem p014) (:domain binding-hubs)
  (:objects a14 a20 - agent
            t5 t25 t30 t33 t49 t52 t88 t92 t93 t104 t129 t130 t140 t146 t161 t163 t175 t176 t181 t195 - tool
            h4 h5 h10 h14 h62 h66 h71 h73 h79 h98 - hub
            x5 x16 - job)
  (:init (end t104 h66) (end t104 h73) (end t129 h4) (end t129 h66) (end t130 h14) (end t130 h73) (end t140 h10) (end t140 h62) (end t146 h5) (end t146 h79) (end t161 h10) (end t161 h4) (end t163 h10) (end t163 h98) (end t175 h66) (end t175 h98) (end t176 h5) (end t176 h62) (end t181 h14) (end t181 h4) (end t195 h62) (end t195 h71) (end t25 h4) (end t25 h71) (end t30 h62) (end t30 h66) (end t33 h5) (end t33 h98) (end t49 h5) (end t49 h73) (end t5 h14) (end t5 h98) (end t52 h71) (end t52 h79) (end t88 h14) (end t88 h79) (end t92 h71) (end t92 h73) (end t93 h10) (end t93 h79) (free t104) (free t129) (free t130) (free t140) (free t146) (free t161) (free t163) (free t175) (free t176) (free t181) (free t195) (free t25) (free t30) (free t33) (free t49) (free t5) (free t52) (free t88) (free t92) (free t93) (link h10 h62) (link h10 h73) (link h14 h4) (link h14 h62) (link h4 h14) (link h4 h5) (link h5 h4) (link h5 h98) (link h62 h10) (link h62 h14) (link h66 h73) (link h66 h79) (link h71 h79) (link h71 h98) (link h73 h10) (link h73 h66) (link h79 h66) (link h79 h71) (link h98 h5) (link h98 h71) (open x16) (open x5) (ready a14) (ready a20))
  (:goal (and (done x16) (done x5))))
