; agents = 2, jobs = 1, hubs = 10
(define (problem p002) (:domain binding-hubs)
  (:objects a10 a13 - agent
            t4 t19 t20 t24 t33 t35 t47 t54 t63 t95 t96 t99 t117 t124 t148 t150 t151 t154 t160 t163 - tool
            h20 h21 h25 h30 h32 h40 h79 h81 h93 h95 - hub
            x9 - job)
  (:init (end t117 h25) (end t117 h81) (end t124 h32) (end t124 h81) (end t148 h20) (end t148 h93) (end t150 h21) (end t150 h95) (end t151 h40) (end t151 h95) (end t154 h30) (end t154 h93) (end t160 h93) (end t160 h95) (end t163 h30) (end t163 h81) (end t19 h20) (end t19 h81) (end t20 h30) (end t20 h40) (end t24 h32) (end t24 h79) (end t33 h20) (end t33 h21) (end t35 h25) (end t35 h40) (end t4 h32) (end t4 h40) (end t47 h79) (end t47 h95) (end t54 h25) (end t54 h93) (end t63 h20) (end t63 h79) (end t95 h25) (end t95 h79) (end t96 h21) (end t96 h32) (end t99 h21) (end t99 h30) (free t117) (free t124) (free t148) (free t150) (free t151) (free t154) (free t160) (free t163) (free t19) (free t20) (free t24) (free t33) (free t35) (free t4) (free t47) (free t54) (free t63) (free t95) (free t96) (free t99) (link h20 h40) (link h20 h81) (link h21 h25) (link h21 h95) (link h25 h21) (link h25 h93) (link h30 h40) (link h30 h79) (link h32 h79) (link h32 h93) (link h40 h20) (link h40 h30) (link h79 h30) (link h79 h32) (link h81 h20) (link h81 h95) (link h93 h25) (link h93 h32) (link h95 h21) (link h95 h81) (open x9) (ready a10) (ready a13))
  (:goal (and (done x9))))
