; agents = 3, jobs = 1, hubs = 8
(define (problem p020) (:domain binding-hubs)
  (:objects a4 a7 a28 - agent
            t14 t32 t33 t36 t40 t68 t75 t92 t94 t107 t108 t114 t117 t136 t150 t154 - tool
            h15 h31 h36 h40 h46 h62 h65 h77 - hub
            x5 - job)
  (:init (end t107 h36) (end t107 h40) (end t108 h62) (end t108 h77) (end t114 h15) (end t114 h77) (end t117 h31) (end t117 h77) (end t136 h40) (end t136 h77) (end t14 h31) (end t14 h62) (end t150 h31) (end t150 h40) (end t154 h46) (end t154 h62) (end t32 h31) (end t32 h65) (end t33 h36) (end t33 h65) (end t36 h46) (end t36 h65) (end t40 h36) (end t40 h46) (end t68 h15) (end t68 h36) (end t75 h15) (end t75 h46) (end t92 h15) (end t92 h65) (end t94 h40) (end t94 h62) (free t107) (free t108) (free t114) (free t117) (free t136) (free t14) (free t150) (free t154) (free t32) (free t33) (free t36) (free t40) (free t68) (free t75) (free t92) (free t94) (link h15 h62) (link h15 h65) (link h31 h36) (link h31 h62) (link h36 h31) (link h36 h46) (link h40 h65) (link h40 h77) (link h46 h36) (link h46 h77) (link h62 h15) (link h62 h31) (link h65 h15) (link h65 h40) (link h77 h40) (link h77 h46) (open x5) (ready a28) (ready a4) (ready a7))
  (:goal (and (done x5))))
