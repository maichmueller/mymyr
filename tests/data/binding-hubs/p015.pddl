; agents = 2, jobs = 2, hubs = 10
(define (problem p015) (:domain binding-hubs)
  (:objects a4 a10 - agent
            t4 t16 t20 t25 t35 t54 t84 t91 t96 t101 t110 t111 t119 t120 t132 t149 t154 t163 t175 t196 - tool
            h6 h24 h26 h47 h48 h52 h53 h59 h90 h95 - hub
            x6 x7 - job)
  (:init (end t101 h53) (end t101 h95) (end t110 h47) (end t110 h52) (end t111 h26) (end t111 h52) (end t119 h48) (end t119 h52) (end t120 h48) (end t120 h59) (end t132 h90) (end t132 h95) (end t149 h6) (end t149 h90) (end t154 h47) (end t154 h6) (end t16 h24) (end t16 h26) (end t163 h52) (end t163 h53) (end t175 h47) (end t175 h95) (end t196 h53) (end t196 h6) (end t20 h24) (end t20 h90) (end t25 h26) (end t25 h59) (end t35 h53) (end t35 h59) (end t4 h24) (end t4 h48) (end t54 h59) (end t54 h90) (end t84 h24) (end t84 h47) (end t91 h48) (end t91 h6) (end t96 h26) (end t96 h95) (free t101) (free t110) (free t111) (free t119) (free t120) (free t132) (free t149) (free t154) (free t16) (free t163) (free t175) (free t196) (free t20) (free t25) (free t35) (free t4) (free t54) (free t84) (free t91) (free t96) (link h24 h26) (link h24 h53) (link h26 h24) (link h26 h6) (link h47 h59) (link h47 h95) (link h48 h52) (link h48 h95) (link h52 h48) (link h52 h90) (link h53 h24) (link h53 h59) (link h59 h47) (link h59 h53) (link h6 h26) (link h6 h90) (link h90 h52) (link h90 h6) (link h95 h47) (link h95 h48) (open x6) (open x7) (ready a10) (ready a4))
  (:goal (and (done x6) (done x7))))
