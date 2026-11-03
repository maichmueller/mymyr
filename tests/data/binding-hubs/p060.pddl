; agents = 4, jobs = 3, hubs = 10
(define (problem p060) (:domain binding-hubs)
  (:objects a18 a24 a27 a40 - agent
            t21 t25 t31 t32 t35 t58 t61 t73 t74 t89 t107 t117 t123 t127 t130 t132 t149 t166 t174 t181 - tool
            h9 h26 h29 h47 h50 h53 h58 h77 h82 h96 - hub
            x10 x11 x14 - job)
  (:init (end t107 h53) (end t107 h58) (end t117 h26) (end t117 h50) (end t123 h47) (end t123 h77) (end t127 h29) (end t127 h9) (end t130 h29) (end t130 h77) (end t132 h26) (end t132 h77) (end t149 h47) (end t149 h58) (end t166 h82) (end t166 h96) (end t174 h26) (end t174 h58) (end t181 h29) (end t181 h96) (end t21 h29) (end t21 h50) (end t25 h47) (end t25 h9) (end t31 h50) (end t31 h82) (end t32 h58) (end t32 h82) (end t35 h53) (end t35 h96) (end t58 h53) (end t58 h9) (end t61 h77) (end t61 h82) (end t73 h26) (end t73 h9) (end t74 h47) (end t74 h96) (end t89 h50) (end t89 h53) (free t107) (free t117) (free t123) (free t127) (free t130) (free t132) (free t149) (free t166) (free t174) (free t181) (free t21) (free t25) (free t31) (free t32) (free t35) (free t58) (free t61) (free t73) (free t74) (free t89) (link h26 h50) (link h26 h96) (link h29 h58) (link h29 h77) (link h47 h50) (link h47 h9) (link h50 h26) (link h50 h47) (link h53 h77) (link h53 h96) (link h58 h29) (link h58 h82) (link h77 h29) (link h77 h53) (link h82 h58) (link h82 h9) (link h9 h47) (link h9 h82) (link h96 h26) (link h96 h53) (open x10) (open x11) (open x14) (ready a18) (ready a24) (ready a27) (ready a40))
  (:goal (and (done x10) (done x11) (done x14))))
