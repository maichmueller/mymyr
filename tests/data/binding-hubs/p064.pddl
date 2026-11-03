; agents = 4, jobs = 3, hubs = 10
(define (problem p064) (:domain binding-hubs)
  (:objects a3 a25 a27 a35 - agent
            t21 t25 t29 t36 t53 t55 t67 t69 t77 t85 t101 t108 t126 t129 t138 t144 t146 t161 t168 t169 - tool
            h11 h29 h34 h46 h48 h50 h55 h62 h77 h95 - hub
            x11 x14 x21 - job)
  (:init (end t101 h34) (end t101 h62) (end t108 h11) (end t108 h34) (end t126 h48) (end t126 h62) (end t129 h11) (end t129 h95) (end t138 h46) (end t138 h50) (end t144 h62) (end t144 h95) (end t146 h34) (end t146 h46) (end t161 h29) (end t161 h50) (end t168 h46) (end t168 h55) (end t169 h48) (end t169 h77) (end t21 h29) (end t21 h34) (end t25 h55) (end t25 h77) (end t29 h29) (end t29 h95) (end t36 h50) (end t36 h77) (end t53 h11) (end t53 h50) (end t55 h11) (end t55 h48) (end t67 h55) (end t67 h62) (end t69 h46) (end t69 h48) (end t77 h77) (end t77 h95) (end t85 h29) (end t85 h55) (free t101) (free t108) (free t126) (free t129) (free t138) (free t144) (free t146) (free t161) (free t168) (free t169) (free t21) (free t25) (free t29) (free t36) (free t53) (free t55) (free t67) (free t69) (free t77) (free t85) (link h11 h48) (link h11 h55) (link h29 h34) (link h29 h48) (link h34 h29) (link h34 h77) (link h46 h50) (link h46 h95) (link h48 h11) (link h48 h29) (link h50 h46) (link h50 h62) (link h55 h11) (link h55 h62) (link h62 h50) (link h62 h55) (link h77 h34) (link h77 h95) (link h95 h46) (link h95 h77) (open x11) (open x14) (open x21) (ready a25) (ready a27) (ready a3) (ready a35))
  (:goal (and (done x11) (done x14) (done x21))))
