; agents = 3, jobs = 2, hubs = 10
(define (problem p025) (:domain binding-hubs)
  (:objects a10 a18 a19 - agent
            t7 t21 t22 t25 t27 t37 t57 t63 t71 t81 t86 t109 t126 t140 t163 t167 t168 t183 t189 t191 - tool
            h18 h43 h44 h45 h51 h59 h64 h67 h87 h98 - hub
            x16 x18 - job)
  (:init (end t109 h45) (end t109 h98) (end t126 h51) (end t126 h64) (end t140 h67) (end t140 h98) (end t163 h51) (end t163 h87) (end t167 h44) (end t167 h64) (end t168 h18) (end t168 h45) (end t183 h44) (end t183 h87) (end t189 h59) (end t189 h98) (end t191 h18) (end t191 h67) (end t21 h43) (end t21 h87) (end t22 h44) (end t22 h59) (end t25 h87) (end t25 h98) (end t27 h51) (end t27 h67) (end t37 h43) (end t37 h64) (end t57 h18) (end t57 h59) (end t63 h43) (end t63 h67) (end t7 h43) (end t7 h59) (end t71 h18) (end t71 h64) (end t81 h44) (end t81 h45) (end t86 h45) (end t86 h51) (free t109) (free t126) (free t140) (free t163) (free t167) (free t168) (free t183) (free t189) (free t191) (free t21) (free t22) (free t25) (free t27) (free t37) (free t57) (free t63) (free t7) (free t71) (free t81) (free t86) (link h18 h64) (link h18 h87) (link h43 h45) (link h43 h87) (link h44 h59) (link h44 h67) (link h45 h43) (link h45 h51) (link h51 h45) (link h51 h59) (link h59 h44) (link h59 h51) (link h64 h18) (link h64 h98) (link h67 h44) (link h67 h98) (link h87 h18) (link h87 h43) (link h98 h64) (link h98 h67) (open x16) (open x18) (ready a10) (ready a18) (ready a19))
  (:goal (and (done x16) (done x18))))
