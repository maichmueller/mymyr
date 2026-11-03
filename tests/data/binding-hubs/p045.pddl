; agents = 4, jobs = 1, hubs = 8
(define (problem p045) (:domain binding-hubs)
  (:objects a27 a28 a31 a35 - agent
            t27 t35 t38 t53 t58 t67 t72 t81 t92 t97 t99 t104 t116 t133 t137 t141 - tool
            h4 h6 h18 h28 h38 h45 h48 h62 - hub
            x10 - job)
  (:init (end t104 h45) (end t104 h62) (end t116 h18) (end t116 h48) (end t133 h28) (end t133 h45) (end t137 h4) (end t137 h48) (end t141 h18) (end t141 h6) (end t27 h28) (end t27 h62) (end t35 h38) (end t35 h48) (end t38 h28) (end t38 h4) (end t53 h18) (end t53 h62) (end t58 h4) (end t58 h45) (end t67 h4) (end t67 h62) (end t72 h38) (end t72 h6) (end t81 h28) (end t81 h38) (end t92 h48) (end t92 h6) (end t97 h45) (end t97 h6) (end t99 h18) (end t99 h38) (free t104) (free t116) (free t133) (free t137) (free t141) (free t27) (free t35) (free t38) (free t53) (free t58) (free t67) (free t72) (free t81) (free t92) (free t97) (free t99) (link h18 h28) (link h18 h6) (link h28 h18) (link h28 h45) (link h38 h48) (link h38 h62) (link h4 h6) (link h4 h62) (link h45 h28) (link h45 h48) (link h48 h38) (link h48 h45) (link h6 h18) (link h6 h4) (link h62 h38) (link h62 h4) (open x10) (ready a27) (ready a28) (ready a31) (ready a35))
  (:goal (and (done x10))))
