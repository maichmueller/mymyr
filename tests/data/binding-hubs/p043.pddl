; agents = 4, jobs = 1, hubs = 8
(define (problem p043) (:domain binding-hubs)
  (:objects a11 a12 a16 a36 - agent
            t19 t47 t53 t55 t79 t99 t103 t108 t115 t120 t121 t123 t125 t126 t139 t153 - tool
            h3 h7 h24 h34 h36 h47 h67 h79 - hub
            x9 - job)
  (:init (end t103 h3) (end t103 h67) (end t108 h24) (end t108 h47) (end t115 h3) (end t115 h34) (end t120 h3) (end t120 h36) (end t121 h24) (end t121 h79) (end t123 h36) (end t123 h7) (end t125 h34) (end t125 h47) (end t126 h67) (end t126 h7) (end t139 h3) (end t139 h7) (end t153 h7) (end t153 h79) (end t19 h34) (end t19 h79) (end t47 h24) (end t47 h36) (end t53 h47) (end t53 h79) (end t55 h24) (end t55 h34) (end t79 h47) (end t79 h67) (end t99 h36) (end t99 h67) (free t103) (free t108) (free t115) (free t120) (free t121) (free t123) (free t125) (free t126) (free t139) (free t153) (free t19) (free t47) (free t53) (free t55) (free t79) (free t99) (link h24 h34) (link h24 h67) (link h3 h67) (link h3 h79) (link h34 h24) (link h34 h7) (link h36 h47) (link h36 h7) (link h47 h36) (link h47 h79) (link h67 h24) (link h67 h3) (link h7 h34) (link h7 h36) (link h79 h3) (link h79 h47) (open x9) (ready a11) (ready a12) (ready a16) (ready a36))
  (:goal (and (done x9))))
