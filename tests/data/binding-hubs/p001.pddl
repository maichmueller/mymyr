; agents = 2, jobs = 1, hubs = 10
(define (problem p001) (:domain binding-hubs)
  (:objects a15 a16 - agent
            t1 t6 t30 t34 t46 t49 t55 t70 t78 t99 t102 t108 t115 t131 t144 t152 t165 t175 t188 t197 - tool
            h6 h13 h19 h24 h27 h28 h51 h74 h78 h83 - hub
            x8 - job)
  (:init (end t1 h27) (end t1 h74) (end t102 h19) (end t102 h24) (end t108 h6) (end t108 h83) (end t115 h19) (end t115 h28) (end t131 h27) (end t131 h78) (end t144 h74) (end t144 h83) (end t152 h51) (end t152 h6) (end t165 h19) (end t165 h74) (end t175 h51) (end t175 h78) (end t188 h24) (end t188 h83) (end t197 h51) (end t197 h74) (end t30 h27) (end t30 h6) (end t34 h13) (end t34 h78) (end t46 h24) (end t46 h27) (end t49 h13) (end t49 h6) (end t55 h28) (end t55 h51) (end t6 h28) (end t6 h83) (end t70 h13) (end t70 h24) (end t78 h19) (end t78 h78) (end t99 h13) (end t99 h28) (free t1) (free t102) (free t108) (free t115) (free t131) (free t144) (free t152) (free t165) (free t175) (free t188) (free t197) (free t30) (free t34) (free t46) (free t49) (free t55) (free t6) (free t70) (free t78) (free t99) (link h13 h6) (link h13 h74) (link h19 h28) (link h19 h6) (link h24 h27) (link h24 h51) (link h27 h24) (link h27 h28) (link h28 h19) (link h28 h27) (link h51 h24) (link h51 h78) (link h6 h13) (link h6 h19) (link h74 h13) (link h74 h83) (link h78 h51) (link h78 h83) (link h83 h74) (link h83 h78) (open x8) (ready a15) (ready a16))
  (:goal (and (done x8))))
