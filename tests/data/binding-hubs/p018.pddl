; agents = 3, jobs = 1, hubs = 8
(define (problem p018) (:domain binding-hubs)
  (:objects a3 a21 a26 - agent
            t12 t15 t41 t44 t73 t99 t103 t108 t110 t121 t127 t135 t139 t141 t153 t157 - tool
            h35 h37 h40 h47 h62 h68 h72 h77 - hub
            x5 - job)
  (:init (end t103 h68) (end t103 h72) (end t108 h37) (end t108 h77) (end t110 h47) (end t110 h72) (end t12 h37) (end t12 h62) (end t121 h37) (end t121 h40) (end t127 h35) (end t127 h68) (end t135 h35) (end t135 h47) (end t139 h37) (end t139 h72) (end t141 h40) (end t141 h77) (end t15 h35) (end t15 h40) (end t153 h47) (end t153 h68) (end t157 h62) (end t157 h77) (end t41 h68) (end t41 h77) (end t44 h47) (end t44 h62) (end t73 h40) (end t73 h62) (end t99 h35) (end t99 h72) (free t103) (free t108) (free t110) (free t12) (free t121) (free t127) (free t135) (free t139) (free t141) (free t15) (free t153) (free t157) (free t41) (free t44) (free t73) (free t99) (link h35 h62) (link h35 h72) (link h37 h62) (link h37 h68) (link h40 h47) (link h40 h77) (link h47 h40) (link h47 h68) (link h62 h35) (link h62 h37) (link h68 h37) (link h68 h47) (link h72 h35) (link h72 h77) (link h77 h40) (link h77 h72) (open x5) (ready a21) (ready a26) (ready a3))
  (:goal (and (done x5))))
