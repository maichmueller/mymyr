; agents = 4, jobs = 3, hubs = 10
(define (problem p061) (:domain binding-hubs)
  (:objects a9 a17 a25 a38 - agent
            t5 t8 t23 t26 t27 t54 t72 t88 t89 t94 t96 t108 t117 t140 t152 t169 t172 t179 t186 t197 - tool
            h3 h10 h20 h22 h47 h60 h61 h65 h82 h86 - hub
            x11 x16 x22 - job)
  (:init (end t108 h3) (end t108 h86) (end t117 h22) (end t117 h61) (end t140 h61) (end t140 h86) (end t152 h20) (end t152 h3) (end t169 h10) (end t169 h82) (end t172 h22) (end t172 h82) (end t179 h20) (end t179 h60) (end t186 h47) (end t186 h86) (end t197 h60) (end t197 h65) (end t23 h47) (end t23 h65) (end t26 h20) (end t26 h47) (end t27 h22) (end t27 h47) (end t5 h60) (end t5 h86) (end t54 h10) (end t54 h60) (end t72 h20) (end t72 h82) (end t8 h10) (end t8 h3) (end t88 h10) (end t88 h61) (end t89 h61) (end t89 h65) (end t94 h22) (end t94 h3) (end t96 h65) (end t96 h82) (free t108) (free t117) (free t140) (free t152) (free t169) (free t172) (free t179) (free t186) (free t197) (free t23) (free t26) (free t27) (free t5) (free t54) (free t72) (free t8) (free t88) (free t89) (free t94) (free t96) (link h10 h47) (link h10 h61) (link h20 h61) (link h20 h82) (link h22 h47) (link h22 h60) (link h3 h65) (link h3 h86) (link h47 h10) (link h47 h22) (link h60 h22) (link h60 h65) (link h61 h10) (link h61 h20) (link h65 h3) (link h65 h60) (link h82 h20) (link h82 h86) (link h86 h3) (link h86 h82) (open x11) (open x16) (open x22) (ready a17) (ready a25) (ready a38) (ready a9))
  (:goal (and (done x11) (done x16) (done x22))))
