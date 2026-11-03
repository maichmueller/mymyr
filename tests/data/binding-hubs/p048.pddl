; agents = 4, jobs = 1, hubs = 8
(define (problem p048) (:domain binding-hubs)
  (:objects a2 a14 a24 a37 - agent
            t10 t13 t17 t18 t20 t35 t71 t98 t99 t106 t107 t116 t119 t139 t157 t159 - tool
            h3 h20 h35 h40 h47 h61 h64 h65 - hub
            x10 - job)
  (:init (end t10 h3) (end t10 h40) (end t106 h64) (end t106 h65) (end t107 h40) (end t107 h61) (end t116 h20) (end t116 h40) (end t119 h3) (end t119 h35) (end t13 h40) (end t13 h64) (end t139 h20) (end t139 h3) (end t157 h35) (end t157 h47) (end t159 h47) (end t159 h61) (end t17 h3) (end t17 h64) (end t18 h61) (end t18 h65) (end t20 h35) (end t20 h65) (end t35 h20) (end t35 h64) (end t71 h20) (end t71 h47) (end t98 h47) (end t98 h65) (end t99 h35) (end t99 h61) (free t10) (free t106) (free t107) (free t116) (free t119) (free t13) (free t139) (free t157) (free t159) (free t17) (free t18) (free t20) (free t35) (free t71) (free t98) (free t99) (link h20 h3) (link h20 h65) (link h3 h20) (link h3 h61) (link h35 h40) (link h35 h65) (link h40 h35) (link h40 h64) (link h47 h61) (link h47 h64) (link h61 h3) (link h61 h47) (link h64 h40) (link h64 h47) (link h65 h20) (link h65 h35) (open x10) (ready a14) (ready a2) (ready a24) (ready a37))
  (:goal (and (done x10))))
