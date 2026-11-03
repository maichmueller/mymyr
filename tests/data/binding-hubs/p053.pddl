; agents = 4, jobs = 2, hubs = 8
(define (problem p053) (:domain binding-hubs)
  (:objects a4 a10 a17 a36 - agent
            t2 t9 t17 t21 t36 t47 t52 t80 t87 t90 t102 t105 t107 t124 t143 t144 - tool
            h8 h25 h28 h32 h33 h36 h38 h71 - hub
            x15 x19 - job)
  (:init (end t102 h38) (end t102 h71) (end t105 h25) (end t105 h8) (end t107 h32) (end t107 h71) (end t124 h25) (end t124 h71) (end t143 h32) (end t143 h33) (end t144 h25) (end t144 h28) (end t17 h28) (end t17 h8) (end t2 h33) (end t2 h71) (end t21 h32) (end t21 h38) (end t36 h33) (end t36 h38) (end t47 h33) (end t47 h8) (end t52 h32) (end t52 h36) (end t80 h25) (end t80 h36) (end t87 h28) (end t87 h36) (end t9 h28) (end t9 h38) (end t90 h36) (end t90 h8) (free t102) (free t105) (free t107) (free t124) (free t143) (free t144) (free t17) (free t2) (free t21) (free t36) (free t47) (free t52) (free t80) (free t87) (free t9) (free t90) (link h25 h36) (link h25 h38) (link h28 h71) (link h28 h8) (link h32 h38) (link h32 h8) (link h33 h36) (link h33 h71) (link h36 h25) (link h36 h33) (link h38 h25) (link h38 h32) (link h71 h28) (link h71 h33) (link h8 h28) (link h8 h32) (open x15) (open x19) (ready a10) (ready a17) (ready a36) (ready a4))
  (:goal (and (done x15) (done x19))))
