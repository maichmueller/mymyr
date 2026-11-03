; agents = 2, jobs = 2, hubs = 8
(define (problem p013) (:domain binding-hubs)
  (:objects a9 a17 - agent
            t6 t35 t41 t51 t53 t56 t97 t106 t112 t136 t137 t138 t139 t151 t157 t158 - tool
            h24 h25 h28 h30 h41 h45 h75 h77 - hub
            x4 x5 - job)
  (:init (end t106 h45) (end t106 h77) (end t112 h25) (end t112 h77) (end t136 h24) (end t136 h41) (end t137 h28) (end t137 h45) (end t138 h75) (end t138 h77) (end t139 h30) (end t139 h77) (end t151 h24) (end t151 h28) (end t157 h25) (end t157 h45) (end t158 h24) (end t158 h25) (end t35 h41) (end t35 h75) (end t41 h28) (end t41 h41) (end t51 h30) (end t51 h45) (end t53 h25) (end t53 h30) (end t56 h24) (end t56 h75) (end t6 h28) (end t6 h75) (end t97 h30) (end t97 h41) (free t106) (free t112) (free t136) (free t137) (free t138) (free t139) (free t151) (free t157) (free t158) (free t35) (free t41) (free t51) (free t53) (free t56) (free t6) (free t97) (link h24 h41) (link h24 h45) (link h25 h28) (link h25 h77) (link h28 h25) (link h28 h75) (link h30 h45) (link h30 h75) (link h41 h24) (link h41 h77) (link h45 h24) (link h45 h30) (link h75 h28) (link h75 h30) (link h77 h25) (link h77 h41) (open x4) (open x5) (ready a17) (ready a9))
  (:goal (and (done x4) (done x5))))
