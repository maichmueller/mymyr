; agents = 3, jobs = 2, hubs = 8
(define (problem p027) (:domain binding-hubs)
  (:objects a16 a19 a30 - agent
            t10 t28 t48 t51 t56 t58 t61 t72 t81 t92 t114 t121 t131 t151 t153 t154 - tool
            h23 h34 h40 h52 h62 h63 h71 h77 - hub
            x11 x13 - job)
  (:init (end t10 h23) (end t10 h52) (end t114 h34) (end t114 h40) (end t121 h23) (end t121 h34) (end t131 h52) (end t131 h77) (end t151 h34) (end t151 h71) (end t153 h40) (end t153 h63) (end t154 h52) (end t154 h62) (end t28 h62) (end t28 h77) (end t48 h52) (end t48 h71) (end t51 h34) (end t51 h63) (end t56 h40) (end t56 h62) (end t58 h23) (end t58 h62) (end t61 h63) (end t61 h77) (end t72 h63) (end t72 h71) (end t81 h40) (end t81 h71) (end t92 h23) (end t92 h77) (free t10) (free t114) (free t121) (free t131) (free t151) (free t153) (free t154) (free t28) (free t48) (free t51) (free t56) (free t58) (free t61) (free t72) (free t81) (free t92) (link h23 h40) (link h23 h52) (link h34 h62) (link h34 h63) (link h40 h23) (link h40 h71) (link h52 h23) (link h52 h63) (link h62 h34) (link h62 h77) (link h63 h34) (link h63 h52) (link h71 h40) (link h71 h77) (link h77 h62) (link h77 h71) (open x11) (open x13) (ready a16) (ready a19) (ready a30))
  (:goal (and (done x11) (done x13))))
