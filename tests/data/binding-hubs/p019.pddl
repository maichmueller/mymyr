; agents = 3, jobs = 1, hubs = 8
(define (problem p019) (:domain binding-hubs)
  (:objects a1 a8 a25 - agent
            t11 t13 t14 t19 t41 t58 t65 t81 t82 t100 t102 t105 t110 t123 t149 t150 - tool
            h6 h13 h15 h19 h32 h60 h67 h80 - hub
            x1 - job)
  (:init (end t100 h13) (end t100 h32) (end t102 h13) (end t102 h80) (end t105 h15) (end t105 h80) (end t11 h13) (end t11 h67) (end t110 h19) (end t110 h6) (end t123 h67) (end t123 h80) (end t13 h15) (end t13 h60) (end t14 h6) (end t14 h60) (end t149 h19) (end t149 h32) (end t150 h32) (end t150 h6) (end t19 h15) (end t19 h67) (end t41 h19) (end t41 h60) (end t58 h32) (end t58 h60) (end t65 h13) (end t65 h15) (end t81 h19) (end t81 h67) (end t82 h6) (end t82 h80) (free t100) (free t102) (free t105) (free t11) (free t110) (free t123) (free t13) (free t14) (free t149) (free t150) (free t19) (free t41) (free t58) (free t65) (free t81) (free t82) (link h13 h60) (link h13 h67) (link h15 h32) (link h15 h80) (link h19 h60) (link h19 h80) (link h32 h15) (link h32 h6) (link h6 h32) (link h6 h67) (link h60 h13) (link h60 h19) (link h67 h13) (link h67 h6) (link h80 h15) (link h80 h19) (open x1) (ready a1) (ready a25) (ready a8))
  (:goal (and (done x1))))
