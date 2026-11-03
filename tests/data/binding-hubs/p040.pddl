; agents = 3, jobs = 3, hubs = 10
(define (problem p040) (:domain binding-hubs)
  (:objects a2 a7 a20 - agent
            t5 t8 t14 t19 t30 t39 t46 t47 t68 t84 t112 t114 t115 t128 t129 t146 t147 t156 t183 t186 - tool
            h7 h19 h31 h40 h55 h70 h75 h80 h82 h93 - hub
            x2 x11 x23 - job)
  (:init (end t112 h40) (end t112 h80) (end t114 h7) (end t114 h82) (end t115 h19) (end t115 h7) (end t128 h31) (end t128 h7) (end t129 h19) (end t129 h55) (end t14 h75) (end t14 h93) (end t146 h31) (end t146 h55) (end t147 h31) (end t147 h80) (end t156 h75) (end t156 h80) (end t183 h70) (end t183 h82) (end t186 h40) (end t186 h70) (end t19 h19) (end t19 h70) (end t30 h7) (end t30 h75) (end t39 h31) (end t39 h70) (end t46 h40) (end t46 h93) (end t47 h40) (end t47 h55) (end t5 h19) (end t5 h93) (end t68 h55) (end t68 h75) (end t8 h80) (end t8 h82) (end t84 h82) (end t84 h93) (free t112) (free t114) (free t115) (free t128) (free t129) (free t14) (free t146) (free t147) (free t156) (free t183) (free t186) (free t19) (free t30) (free t39) (free t46) (free t47) (free t5) (free t68) (free t8) (free t84) (link h19 h7) (link h19 h80) (link h31 h80) (link h31 h93) (link h40 h55) (link h40 h7) (link h55 h40) (link h55 h82) (link h7 h19) (link h7 h40) (link h70 h75) (link h70 h82) (link h75 h70) (link h75 h93) (link h80 h19) (link h80 h31) (link h82 h55) (link h82 h70) (link h93 h31) (link h93 h75) (open x11) (open x2) (open x23) (ready a2) (ready a20) (ready a7))
  (:goal (and (done x11) (done x2) (done x23))))
