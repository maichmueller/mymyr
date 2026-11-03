; agents = 3, jobs = 1, hubs = 10
(define (problem p024) (:domain binding-hubs)
  (:objects a24 a29 a30 - agent
            t7 t22 t23 t40 t46 t50 t64 t93 t94 t95 t100 t102 t133 t139 t145 t150 t152 t174 t190 t195 - tool
            h3 h8 h21 h29 h37 h38 h59 h67 h70 h73 - hub
            x1 - job)
  (:init (end t100 h21) (end t100 h67) (end t102 h3) (end t102 h8) (end t133 h37) (end t133 h59) (end t139 h3) (end t139 h38) (end t145 h21) (end t145 h59) (end t150 h29) (end t150 h70) (end t152 h37) (end t152 h70) (end t174 h37) (end t174 h67) (end t190 h38) (end t190 h70) (end t195 h38) (end t195 h67) (end t22 h67) (end t22 h8) (end t23 h29) (end t23 h73) (end t40 h59) (end t40 h8) (end t46 h21) (end t46 h3) (end t50 h73) (end t50 h8) (end t64 h29) (end t64 h3) (end t7 h29) (end t7 h59) (end t93 h21) (end t93 h70) (end t94 h38) (end t94 h73) (end t95 h37) (end t95 h73) (free t100) (free t102) (free t133) (free t139) (free t145) (free t150) (free t152) (free t174) (free t190) (free t195) (free t22) (free t23) (free t40) (free t46) (free t50) (free t64) (free t7) (free t93) (free t94) (free t95) (link h21 h70) (link h21 h73) (link h29 h3) (link h29 h67) (link h3 h29) (link h3 h37) (link h37 h3) (link h37 h59) (link h38 h59) (link h38 h73) (link h59 h37) (link h59 h38) (link h67 h29) (link h67 h8) (link h70 h21) (link h70 h8) (link h73 h21) (link h73 h38) (link h8 h67) (link h8 h70) (open x1) (ready a24) (ready a29) (ready a30))
  (:goal (and (done x1))))
