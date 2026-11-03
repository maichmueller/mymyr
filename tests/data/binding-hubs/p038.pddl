; agents = 3, jobs = 3, hubs = 10
(define (problem p038) (:domain binding-hubs)
  (:objects a6 a9 a24 - agent
            t24 t30 t34 t47 t58 t59 t61 t70 t71 t76 t78 t96 t119 t129 t133 t135 t152 t164 t170 t171 - tool
            h1 h6 h10 h33 h37 h57 h68 h69 h76 h77 - hub
            x14 x19 x26 - job)
  (:init (end t119 h1) (end t119 h33) (end t129 h6) (end t129 h68) (end t133 h69) (end t133 h77) (end t135 h68) (end t135 h77) (end t152 h1) (end t152 h37) (end t164 h37) (end t164 h68) (end t170 h1) (end t170 h57) (end t171 h1) (end t171 h77) (end t24 h6) (end t24 h69) (end t30 h6) (end t30 h76) (end t34 h57) (end t34 h76) (end t47 h57) (end t47 h69) (end t58 h33) (end t58 h68) (end t59 h33) (end t59 h76) (end t61 h10) (end t61 h57) (end t70 h33) (end t70 h69) (end t71 h10) (end t71 h77) (end t76 h37) (end t76 h76) (end t78 h10) (end t78 h37) (end t96 h10) (end t96 h6) (free t119) (free t129) (free t133) (free t135) (free t152) (free t164) (free t170) (free t171) (free t24) (free t30) (free t34) (free t47) (free t58) (free t59) (free t61) (free t70) (free t71) (free t76) (free t78) (free t96) (link h1 h6) (link h1 h77) (link h10 h33) (link h10 h57) (link h33 h10) (link h33 h69) (link h37 h69) (link h37 h76) (link h57 h10) (link h57 h68) (link h6 h1) (link h6 h68) (link h68 h57) (link h68 h6) (link h69 h33) (link h69 h37) (link h76 h37) (link h76 h77) (link h77 h1) (link h77 h76) (open x14) (open x19) (open x26) (ready a24) (ready a6) (ready a9))
  (:goal (and (done x14) (done x19) (done x26))))
