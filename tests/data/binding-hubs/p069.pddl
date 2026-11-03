; agents = 4, jobs = 4, hubs = 8
(define (problem p069) (:domain binding-hubs)
  (:objects a7 a16 a20 a23 - agent
            t28 t31 t40 t42 t44 t49 t52 t56 t75 t84 t89 t94 t95 t137 t150 t157 - tool
            h3 h15 h23 h32 h34 h46 h69 h78 - hub
            x9 x17 x19 x36 - job)
  (:init (end t137 h23) (end t137 h69) (end t150 h3) (end t150 h78) (end t157 h32) (end t157 h78) (end t28 h15) (end t28 h23) (end t31 h15) (end t31 h3) (end t40 h34) (end t40 h78) (end t42 h23) (end t42 h32) (end t44 h23) (end t44 h34) (end t49 h46) (end t49 h69) (end t52 h3) (end t52 h46) (end t56 h15) (end t56 h32) (end t75 h32) (end t75 h69) (end t84 h46) (end t84 h78) (end t89 h34) (end t89 h46) (end t94 h3) (end t94 h34) (end t95 h15) (end t95 h69) (free t137) (free t150) (free t157) (free t28) (free t31) (free t40) (free t42) (free t44) (free t49) (free t52) (free t56) (free t75) (free t84) (free t89) (free t94) (free t95) (link h15 h32) (link h15 h34) (link h23 h3) (link h23 h69) (link h3 h23) (link h3 h46) (link h32 h15) (link h32 h46) (link h34 h15) (link h34 h78) (link h46 h3) (link h46 h32) (link h69 h23) (link h69 h78) (link h78 h34) (link h78 h69) (open x17) (open x19) (open x36) (open x9) (ready a16) (ready a20) (ready a23) (ready a7))
  (:goal (and (done x17) (done x19) (done x36) (done x9))))
