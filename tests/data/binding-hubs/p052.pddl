; agents = 4, jobs = 2, hubs = 8
(define (problem p052) (:domain binding-hubs)
  (:objects a15 a19 a26 a40 - agent
            t28 t31 t37 t38 t45 t56 t65 t73 t100 t105 t111 t114 t125 t140 t143 t148 - tool
            h3 h19 h20 h35 h36 h42 h73 h76 - hub
            x9 x13 - job)
  (:init (end t100 h19) (end t100 h36) (end t105 h20) (end t105 h42) (end t111 h19) (end t111 h3) (end t114 h36) (end t114 h73) (end t125 h42) (end t125 h76) (end t140 h42) (end t140 h73) (end t143 h3) (end t143 h36) (end t148 h73) (end t148 h76) (end t28 h19) (end t28 h42) (end t31 h35) (end t31 h36) (end t37 h3) (end t37 h35) (end t38 h19) (end t38 h35) (end t45 h20) (end t45 h76) (end t56 h20) (end t56 h35) (end t65 h20) (end t65 h73) (end t73 h3) (end t73 h76) (free t100) (free t105) (free t111) (free t114) (free t125) (free t140) (free t143) (free t148) (free t28) (free t31) (free t37) (free t38) (free t45) (free t56) (free t65) (free t73) (link h19 h36) (link h19 h76) (link h20 h36) (link h20 h42) (link h3 h35) (link h3 h42) (link h35 h3) (link h35 h73) (link h36 h19) (link h36 h20) (link h42 h20) (link h42 h3) (link h73 h35) (link h73 h76) (link h76 h19) (link h76 h73) (open x13) (open x9) (ready a15) (ready a19) (ready a26) (ready a40))
  (:goal (and (done x13) (done x9))))
