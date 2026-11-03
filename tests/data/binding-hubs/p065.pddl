; agents = 4, jobs = 4, hubs = 8
(define (problem p065) (:domain binding-hubs)
  (:objects a15 a21 a26 a36 - agent
            t2 t3 t9 t26 t27 t59 t81 t92 t93 t96 t105 t124 t125 t134 t135 t152 - tool
            h8 h27 h32 h39 h41 h47 h48 h77 - hub
            x8 x12 x32 x34 - job)
  (:init (end t105 h41) (end t105 h77) (end t124 h39) (end t124 h47) (end t125 h27) (end t125 h32) (end t134 h48) (end t134 h8) (end t135 h39) (end t135 h41) (end t152 h32) (end t152 h39) (end t2 h77) (end t2 h8) (end t26 h32) (end t26 h47) (end t27 h27) (end t27 h47) (end t3 h47) (end t3 h8) (end t59 h27) (end t59 h77) (end t81 h41) (end t81 h8) (end t9 h27) (end t9 h39) (end t92 h41) (end t92 h48) (end t93 h32) (end t93 h48) (end t96 h48) (end t96 h77) (free t105) (free t124) (free t125) (free t134) (free t135) (free t152) (free t2) (free t26) (free t27) (free t3) (free t59) (free t81) (free t9) (free t92) (free t93) (free t96) (link h27 h39) (link h27 h8) (link h32 h41) (link h32 h47) (link h39 h27) (link h39 h48) (link h41 h32) (link h41 h8) (link h47 h32) (link h47 h77) (link h48 h39) (link h48 h77) (link h77 h47) (link h77 h48) (link h8 h27) (link h8 h41) (open x12) (open x32) (open x34) (open x8) (ready a15) (ready a21) (ready a26) (ready a36))
  (:goal (and (done x12) (done x32) (done x34) (done x8))))
