; agents = 3, jobs = 2, hubs = 8
(define (problem p031) (:domain binding-hubs)
  (:objects a9 a16 a19 - agent
            t10 t24 t25 t27 t34 t37 t46 t47 t50 t85 t98 t103 t115 t119 t152 t160 - tool
            h5 h12 h13 h16 h21 h30 h53 h56 - hub
            x3 x11 - job)
  (:init (end t10 h13) (end t10 h16) (end t103 h12) (end t103 h21) (end t115 h30) (end t115 h5) (end t119 h30) (end t119 h56) (end t152 h5) (end t152 h53) (end t160 h16) (end t160 h30) (end t24 h53) (end t24 h56) (end t25 h5) (end t25 h56) (end t27 h12) (end t27 h5) (end t34 h21) (end t34 h53) (end t37 h13) (end t37 h21) (end t46 h12) (end t46 h16) (end t47 h13) (end t47 h56) (end t50 h30) (end t50 h53) (end t85 h12) (end t85 h13) (end t98 h16) (end t98 h21) (free t10) (free t103) (free t115) (free t119) (free t152) (free t160) (free t24) (free t25) (free t27) (free t34) (free t37) (free t46) (free t47) (free t50) (free t85) (free t98) (link h12 h13) (link h12 h53) (link h13 h12) (link h13 h30) (link h16 h21) (link h16 h56) (link h21 h16) (link h21 h5) (link h30 h13) (link h30 h5) (link h5 h21) (link h5 h30) (link h53 h12) (link h53 h56) (link h56 h16) (link h56 h53) (open x11) (open x3) (ready a16) (ready a19) (ready a9))
  (:goal (and (done x11) (done x3))))
