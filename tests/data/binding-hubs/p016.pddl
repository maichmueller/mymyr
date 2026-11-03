; agents = 2, jobs = 2, hubs = 8
(define (problem p016) (:domain binding-hubs)
  (:objects a7 a19 - agent
            t30 t35 t41 t47 t48 t53 t67 t72 t85 t88 t95 t104 t111 t130 t138 t150 - tool
            h2 h7 h10 h13 h39 h44 h46 h61 - hub
            x8 x9 - job)
  (:init (end t104 h46) (end t104 h61) (end t111 h10) (end t111 h13) (end t130 h10) (end t130 h44) (end t138 h44) (end t138 h46) (end t150 h39) (end t150 h61) (end t30 h13) (end t30 h44) (end t35 h13) (end t35 h7) (end t41 h2) (end t41 h39) (end t47 h2) (end t47 h44) (end t48 h61) (end t48 h7) (end t53 h39) (end t53 h46) (end t67 h46) (end t67 h7) (end t72 h10) (end t72 h61) (end t85 h10) (end t85 h2) (end t88 h13) (end t88 h2) (end t95 h39) (end t95 h7) (free t104) (free t111) (free t130) (free t138) (free t150) (free t30) (free t35) (free t41) (free t47) (free t48) (free t53) (free t67) (free t72) (free t85) (free t88) (free t95) (link h10 h44) (link h10 h7) (link h13 h2) (link h13 h61) (link h2 h13) (link h2 h46) (link h39 h44) (link h39 h61) (link h44 h10) (link h44 h39) (link h46 h2) (link h46 h7) (link h61 h13) (link h61 h39) (link h7 h10) (link h7 h46) (open x8) (open x9) (ready a19) (ready a7))
  (:goal (and (done x8) (done x9))))
