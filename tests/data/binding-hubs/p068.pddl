; agents = 4, jobs = 4, hubs = 8
(define (problem p068) (:domain binding-hubs)
  (:objects a10 a26 a31 a38 - agent
            t12 t17 t24 t55 t60 t63 t66 t67 t84 t90 t97 t106 t111 t126 t148 t159 - tool
            h9 h24 h50 h53 h59 h63 h68 h72 - hub
            x8 x12 x15 x40 - job)
  (:init (end t106 h24) (end t106 h59) (end t111 h24) (end t111 h50) (end t12 h63) (end t12 h68) (end t126 h59) (end t126 h68) (end t148 h50) (end t148 h53) (end t159 h53) (end t159 h63) (end t17 h50) (end t17 h72) (end t24 h24) (end t24 h53) (end t55 h59) (end t55 h72) (end t60 h50) (end t60 h63) (end t63 h68) (end t63 h9) (end t66 h59) (end t66 h9) (end t67 h53) (end t67 h9) (end t84 h72) (end t84 h9) (end t90 h68) (end t90 h72) (end t97 h24) (end t97 h63) (free t106) (free t111) (free t12) (free t126) (free t148) (free t159) (free t17) (free t24) (free t55) (free t60) (free t63) (free t66) (free t67) (free t84) (free t90) (free t97) (link h24 h53) (link h24 h72) (link h50 h59) (link h50 h63) (link h53 h24) (link h53 h68) (link h59 h50) (link h59 h68) (link h63 h50) (link h63 h9) (link h68 h53) (link h68 h59) (link h72 h24) (link h72 h9) (link h9 h63) (link h9 h72) (open x12) (open x15) (open x40) (open x8) (ready a10) (ready a26) (ready a31) (ready a38))
  (:goal (and (done x12) (done x15) (done x40) (done x8))))
