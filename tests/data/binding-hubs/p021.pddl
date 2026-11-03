; agents = 3, jobs = 1, hubs = 8
(define (problem p021) (:domain binding-hubs)
  (:objects a7 a8 a25 - agent
            t3 t7 t10 t11 t16 t27 t51 t57 t64 t67 t70 t72 t109 t129 t139 t160 - tool
            h10 h12 h21 h31 h48 h62 h71 h74 - hub
            x6 - job)
  (:init (end t10 h21) (end t10 h74) (end t109 h31) (end t109 h74) (end t11 h62) (end t11 h74) (end t129 h21) (end t129 h71) (end t139 h21) (end t139 h62) (end t16 h10) (end t16 h71) (end t160 h12) (end t160 h62) (end t27 h10) (end t27 h31) (end t3 h12) (end t3 h31) (end t51 h31) (end t51 h48) (end t57 h12) (end t57 h48) (end t64 h10) (end t64 h12) (end t67 h62) (end t67 h71) (end t7 h10) (end t7 h48) (end t70 h21) (end t70 h48) (end t72 h71) (end t72 h74) (free t10) (free t109) (free t11) (free t129) (free t139) (free t16) (free t160) (free t27) (free t3) (free t51) (free t57) (free t64) (free t67) (free t7) (free t70) (free t72) (link h10 h12) (link h10 h74) (link h12 h10) (link h12 h21) (link h21 h12) (link h21 h71) (link h31 h48) (link h31 h71) (link h48 h31) (link h48 h62) (link h62 h48) (link h62 h74) (link h71 h21) (link h71 h31) (link h74 h10) (link h74 h62) (open x6) (ready a25) (ready a7) (ready a8))
  (:goal (and (done x6))))
