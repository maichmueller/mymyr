; agents = 3, jobs = 2, hubs = 8
(define (problem p028) (:domain binding-hubs)
  (:objects a5 a7 a26 - agent
            t3 t15 t41 t61 t64 t71 t83 t96 t99 t107 t108 t116 t121 t133 t136 t156 - tool
            h7 h17 h19 h34 h35 h67 h69 h78 - hub
            x5 x19 - job)
  (:init (end t107 h67) (end t107 h69) (end t108 h67) (end t108 h7) (end t116 h19) (end t116 h78) (end t121 h17) (end t121 h69) (end t133 h35) (end t133 h78) (end t136 h17) (end t136 h67) (end t15 h17) (end t15 h19) (end t156 h19) (end t156 h35) (end t3 h67) (end t3 h78) (end t41 h34) (end t41 h78) (end t61 h34) (end t61 h35) (end t64 h34) (end t64 h69) (end t71 h69) (end t71 h7) (end t83 h17) (end t83 h7) (end t96 h35) (end t96 h7) (end t99 h19) (end t99 h34) (free t107) (free t108) (free t116) (free t121) (free t133) (free t136) (free t15) (free t156) (free t3) (free t41) (free t61) (free t64) (free t71) (free t83) (free t96) (free t99) (link h17 h35) (link h17 h67) (link h19 h34) (link h19 h7) (link h34 h19) (link h34 h67) (link h35 h17) (link h35 h78) (link h67 h17) (link h67 h34) (link h69 h7) (link h69 h78) (link h7 h19) (link h7 h69) (link h78 h35) (link h78 h69) (open x19) (open x5) (ready a26) (ready a5) (ready a7))
  (:goal (and (done x19) (done x5))))
