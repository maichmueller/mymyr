; agents = 2, jobs = 2, hubs = 10
(define (problem p009) (:domain binding-hubs)
  (:objects a1 a6 - agent
            t10 t23 t41 t46 t58 t70 t89 t101 t106 t121 t129 t134 t137 t139 t158 t179 t180 t183 t194 t200 - tool
            h1 h17 h26 h49 h50 h58 h59 h71 h81 h93 - hub
            x17 x19 - job)
  (:init (end t10 h49) (end t10 h71) (end t101 h50) (end t101 h81) (end t106 h59) (end t106 h71) (end t121 h17) (end t121 h58) (end t129 h49) (end t129 h50) (end t134 h1) (end t134 h93) (end t137 h26) (end t137 h59) (end t139 h26) (end t139 h49) (end t158 h17) (end t158 h71) (end t179 h71) (end t179 h81) (end t180 h1) (end t180 h17) (end t183 h1) (end t183 h81) (end t194 h50) (end t194 h93) (end t200 h58) (end t200 h81) (end t23 h49) (end t23 h58) (end t41 h26) (end t41 h93) (end t46 h58) (end t46 h93) (end t58 h50) (end t58 h59) (end t70 h17) (end t70 h26) (end t89 h1) (end t89 h59) (free t10) (free t101) (free t106) (free t121) (free t129) (free t134) (free t137) (free t139) (free t158) (free t179) (free t180) (free t183) (free t194) (free t200) (free t23) (free t41) (free t46) (free t58) (free t70) (free t89) (link h1 h49) (link h1 h81) (link h17 h50) (link h17 h58) (link h26 h59) (link h26 h81) (link h49 h1) (link h49 h71) (link h50 h17) (link h50 h93) (link h58 h17) (link h58 h59) (link h59 h26) (link h59 h58) (link h71 h49) (link h71 h93) (link h81 h1) (link h81 h26) (link h93 h50) (link h93 h71) (open x17) (open x19) (ready a1) (ready a6))
  (:goal (and (done x17) (done x19))))
