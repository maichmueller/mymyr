; agents = 4, jobs = 2, hubs = 10
(define (problem p056) (:domain binding-hubs)
  (:objects a15 a27 a31 a32 - agent
            t3 t15 t18 t38 t46 t47 t55 t70 t83 t92 t97 t100 t135 t163 t165 t170 t174 t181 t187 t200 - tool
            h15 h16 h45 h55 h57 h65 h73 h79 h81 h92 - hub
            x14 x15 - job)
  (:init (end t100 h45) (end t100 h81) (end t135 h15) (end t135 h55) (end t15 h15) (end t15 h92) (end t163 h16) (end t163 h57) (end t165 h57) (end t165 h73) (end t170 h55) (end t170 h79) (end t174 h79) (end t174 h92) (end t18 h15) (end t18 h16) (end t181 h57) (end t181 h92) (end t187 h15) (end t187 h45) (end t200 h45) (end t200 h79) (end t3 h16) (end t3 h65) (end t38 h73) (end t38 h79) (end t46 h55) (end t46 h81) (end t47 h65) (end t47 h92) (end t55 h16) (end t55 h81) (end t70 h65) (end t70 h73) (end t83 h55) (end t83 h57) (end t92 h73) (end t92 h81) (end t97 h45) (end t97 h65) (free t100) (free t135) (free t15) (free t163) (free t165) (free t170) (free t174) (free t18) (free t181) (free t187) (free t200) (free t3) (free t38) (free t46) (free t47) (free t55) (free t70) (free t83) (free t92) (free t97) (link h15 h16) (link h15 h73) (link h16 h15) (link h16 h79) (link h45 h57) (link h45 h79) (link h55 h65) (link h55 h81) (link h57 h45) (link h57 h92) (link h65 h55) (link h65 h73) (link h73 h15) (link h73 h65) (link h79 h16) (link h79 h45) (link h81 h55) (link h81 h92) (link h92 h57) (link h92 h81) (open x14) (open x15) (ready a15) (ready a27) (ready a31) (ready a32))
  (:goal (and (done x14) (done x15))))
