; agents = 3, jobs = 2, hubs = 10
(define (problem p029) (:domain binding-hubs)
  (:objects a9 a12 a20 - agent
            t5 t22 t24 t25 t49 t50 t54 t78 t107 t113 t120 t123 t130 t136 t140 t151 t175 t178 t184 t192 - tool
            h5 h10 h21 h40 h41 h52 h53 h64 h68 h70 - hub
            x10 x18 - job)
  (:init (end t107 h10) (end t107 h53) (end t113 h52) (end t113 h68) (end t120 h5) (end t120 h68) (end t123 h40) (end t123 h41) (end t130 h21) (end t130 h52) (end t136 h53) (end t136 h68) (end t140 h41) (end t140 h64) (end t151 h40) (end t151 h68) (end t175 h40) (end t175 h70) (end t178 h10) (end t178 h52) (end t184 h5) (end t184 h70) (end t192 h21) (end t192 h53) (end t22 h64) (end t22 h70) (end t24 h41) (end t24 h53) (end t25 h21) (end t25 h64) (end t49 h52) (end t49 h70) (end t5 h41) (end t5 h5) (end t50 h21) (end t50 h5) (end t54 h10) (end t54 h64) (end t78 h10) (end t78 h40) (free t107) (free t113) (free t120) (free t123) (free t130) (free t136) (free t140) (free t151) (free t175) (free t178) (free t184) (free t192) (free t22) (free t24) (free t25) (free t49) (free t5) (free t50) (free t54) (free t78) (link h10 h5) (link h10 h52) (link h21 h40) (link h21 h64) (link h40 h21) (link h40 h41) (link h41 h40) (link h41 h52) (link h5 h10) (link h5 h70) (link h52 h10) (link h52 h41) (link h53 h68) (link h53 h70) (link h64 h21) (link h64 h68) (link h68 h53) (link h68 h64) (link h70 h5) (link h70 h53) (open x10) (open x18) (ready a12) (ready a20) (ready a9))
  (:goal (and (done x10) (done x18))))
