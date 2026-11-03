; agents = 2, jobs = 1, hubs = 8
(define (problem p004) (:domain binding-hubs)
  (:objects a2 a8 - agent
            t9 t12 t24 t34 t36 t46 t73 t85 t93 t95 t116 t134 t136 t147 t150 t152 - tool
            h3 h5 h10 h40 h46 h61 h62 h80 - hub
            x2 - job)
  (:init (end t116 h10) (end t116 h61) (end t12 h3) (end t12 h61) (end t134 h3) (end t134 h5) (end t136 h3) (end t136 h40) (end t147 h5) (end t147 h61) (end t150 h61) (end t150 h80) (end t152 h10) (end t152 h46) (end t24 h10) (end t24 h40) (end t34 h46) (end t34 h80) (end t36 h3) (end t36 h80) (end t46 h40) (end t46 h46) (end t73 h5) (end t73 h80) (end t85 h46) (end t85 h62) (end t9 h40) (end t9 h62) (end t93 h5) (end t93 h62) (end t95 h10) (end t95 h62) (free t116) (free t12) (free t134) (free t136) (free t147) (free t150) (free t152) (free t24) (free t34) (free t36) (free t46) (free t73) (free t85) (free t9) (free t93) (free t95) (link h10 h62) (link h10 h80) (link h3 h61) (link h3 h62) (link h40 h46) (link h40 h5) (link h46 h40) (link h46 h61) (link h5 h40) (link h5 h80) (link h61 h3) (link h61 h46) (link h62 h10) (link h62 h3) (link h80 h10) (link h80 h5) (open x2) (ready a2) (ready a8))
  (:goal (and (done x2))))
