; agents = 3, jobs = 1, hubs = 8
(define (problem p023) (:domain binding-hubs)
  (:objects a15 a22 a27 - agent
            t3 t20 t21 t31 t37 t55 t57 t73 t77 t78 t80 t84 t92 t94 t127 t155 - tool
            h36 h40 h42 h57 h68 h73 h77 h79 - hub
            x8 - job)
  (:init (end t127 h36) (end t127 h40) (end t155 h36) (end t155 h57) (end t20 h36) (end t20 h79) (end t21 h73) (end t21 h79) (end t3 h73) (end t3 h77) (end t31 h40) (end t31 h77) (end t37 h40) (end t37 h73) (end t55 h42) (end t55 h77) (end t57 h36) (end t57 h42) (end t73 h57) (end t73 h68) (end t77 h42) (end t77 h57) (end t78 h68) (end t78 h77) (end t80 h40) (end t80 h68) (end t84 h42) (end t84 h79) (end t92 h57) (end t92 h79) (end t94 h68) (end t94 h73) (free t127) (free t155) (free t20) (free t21) (free t3) (free t31) (free t37) (free t55) (free t57) (free t73) (free t77) (free t78) (free t80) (free t84) (free t92) (free t94) (link h36 h77) (link h36 h79) (link h40 h42) (link h40 h68) (link h42 h40) (link h42 h57) (link h57 h42) (link h57 h73) (link h68 h40) (link h68 h79) (link h73 h57) (link h73 h77) (link h77 h36) (link h77 h73) (link h79 h36) (link h79 h68) (open x8) (ready a15) (ready a22) (ready a27))
  (:goal (and (done x8))))
