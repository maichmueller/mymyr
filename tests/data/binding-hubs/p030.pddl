; agents = 3, jobs = 2, hubs = 10
(define (problem p030) (:domain binding-hubs)
  (:objects a13 a15 a22 - agent
            t3 t4 t30 t32 t39 t53 t61 t76 t97 t114 t118 t122 t145 t158 t168 t182 t183 t185 t192 t198 - tool
            h38 h39 h44 h49 h51 h70 h72 h81 h96 h97 - hub
            x3 x9 - job)
  (:init (end t114 h44) (end t114 h97) (end t118 h51) (end t118 h81) (end t122 h44) (end t122 h70) (end t145 h44) (end t145 h72) (end t158 h51) (end t158 h70) (end t168 h39) (end t168 h96) (end t182 h39) (end t182 h49) (end t183 h81) (end t183 h96) (end t185 h44) (end t185 h81) (end t192 h38) (end t192 h81) (end t198 h49) (end t198 h72) (end t3 h38) (end t3 h39) (end t30 h38) (end t30 h97) (end t32 h49) (end t32 h70) (end t39 h72) (end t39 h96) (end t4 h51) (end t4 h72) (end t53 h49) (end t53 h97) (end t61 h96) (end t61 h97) (end t76 h39) (end t76 h51) (end t97 h38) (end t97 h70) (free t114) (free t118) (free t122) (free t145) (free t158) (free t168) (free t182) (free t183) (free t185) (free t192) (free t198) (free t3) (free t30) (free t32) (free t39) (free t4) (free t53) (free t61) (free t76) (free t97) (link h38 h72) (link h38 h81) (link h39 h44) (link h39 h96) (link h44 h39) (link h44 h97) (link h49 h70) (link h49 h81) (link h51 h72) (link h51 h97) (link h70 h49) (link h70 h96) (link h72 h38) (link h72 h51) (link h81 h38) (link h81 h49) (link h96 h39) (link h96 h70) (link h97 h44) (link h97 h51) (open x3) (open x9) (ready a13) (ready a15) (ready a22))
  (:goal (and (done x3) (done x9))))
