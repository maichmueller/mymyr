; agents = 4, jobs = 2, hubs = 8
(define (problem p051) (:domain binding-hubs)
  (:objects a9 a13 a25 a40 - agent
            t4 t13 t35 t36 t41 t48 t60 t65 t69 t70 t75 t99 t102 t108 t121 t150 - tool
            h6 h16 h25 h37 h46 h52 h69 h80 - hub
            x1 x11 - job)
  (:init (end t102 h25) (end t102 h52) (end t108 h16) (end t108 h69) (end t121 h16) (end t121 h37) (end t13 h25) (end t13 h80) (end t150 h46) (end t150 h6) (end t35 h25) (end t35 h69) (end t36 h6) (end t36 h69) (end t4 h52) (end t4 h80) (end t41 h37) (end t41 h52) (end t48 h46) (end t48 h69) (end t60 h25) (end t60 h37) (end t65 h6) (end t65 h80) (end t69 h37) (end t69 h80) (end t70 h46) (end t70 h52) (end t75 h16) (end t75 h6) (end t99 h16) (end t99 h46) (free t102) (free t108) (free t121) (free t13) (free t150) (free t35) (free t36) (free t4) (free t41) (free t48) (free t60) (free t65) (free t69) (free t70) (free t75) (free t99) (link h16 h52) (link h16 h6) (link h25 h37) (link h25 h6) (link h37 h25) (link h37 h46) (link h46 h37) (link h46 h69) (link h52 h16) (link h52 h80) (link h6 h16) (link h6 h25) (link h69 h46) (link h69 h80) (link h80 h52) (link h80 h69) (open x1) (open x11) (ready a13) (ready a25) (ready a40) (ready a9))
  (:goal (and (done x1) (done x11))))
