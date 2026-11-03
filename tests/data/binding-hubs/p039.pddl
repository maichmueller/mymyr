; agents = 3, jobs = 3, hubs = 8
(define (problem p039) (:domain binding-hubs)
  (:objects a6 a26 a29 - agent
            t1 t9 t16 t17 t39 t47 t83 t93 t99 t119 t124 t128 t133 t143 t150 t160 - tool
            h15 h22 h23 h38 h56 h65 h67 h69 - hub
            x8 x15 x22 - job)
  (:init (end t1 h22) (end t1 h67) (end t119 h15) (end t119 h56) (end t124 h15) (end t124 h67) (end t128 h15) (end t128 h69) (end t133 h56) (end t133 h69) (end t143 h56) (end t143 h67) (end t150 h38) (end t150 h69) (end t16 h23) (end t16 h56) (end t160 h22) (end t160 h23) (end t17 h15) (end t17 h65) (end t39 h38) (end t39 h65) (end t47 h22) (end t47 h38) (end t83 h22) (end t83 h65) (end t9 h67) (end t9 h69) (end t93 h23) (end t93 h38) (end t99 h23) (end t99 h65) (free t1) (free t119) (free t124) (free t128) (free t133) (free t143) (free t150) (free t16) (free t160) (free t17) (free t39) (free t47) (free t83) (free t9) (free t93) (free t99) (link h15 h22) (link h15 h56) (link h22 h15) (link h22 h23) (link h23 h22) (link h23 h69) (link h38 h56) (link h38 h65) (link h56 h15) (link h56 h38) (link h65 h38) (link h65 h67) (link h67 h65) (link h67 h69) (link h69 h23) (link h69 h67) (open x15) (open x22) (open x8) (ready a26) (ready a29) (ready a6))
  (:goal (and (done x15) (done x22) (done x8))))
