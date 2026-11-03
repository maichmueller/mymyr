; agents = 4, jobs = 4, hubs = 10
(define (problem p066) (:domain binding-hubs)
  (:objects a6 a7 a26 a37 - agent
            t9 t13 t15 t22 t36 t39 t40 t58 t65 t73 t74 t82 t85 t88 t102 t126 t142 t160 t165 t196 - tool
            h1 h3 h28 h70 h75 h79 h82 h88 h92 h97 - hub
            x5 x22 x27 x31 - job)
  (:init (end t102 h1) (end t102 h3) (end t126 h75) (end t126 h82) (end t13 h70) (end t13 h92) (end t142 h3) (end t142 h75) (end t15 h70) (end t15 h97) (end t160 h82) (end t160 h97) (end t165 h28) (end t165 h79) (end t196 h1) (end t196 h28) (end t22 h28) (end t22 h92) (end t36 h70) (end t36 h79) (end t39 h79) (end t39 h82) (end t40 h88) (end t40 h92) (end t58 h82) (end t58 h92) (end t65 h3) (end t65 h79) (end t73 h75) (end t73 h88) (end t74 h3) (end t74 h97) (end t82 h1) (end t82 h70) (end t85 h88) (end t85 h97) (end t88 h1) (end t88 h88) (end t9 h28) (end t9 h75) (free t102) (free t126) (free t13) (free t142) (free t15) (free t160) (free t165) (free t196) (free t22) (free t36) (free t39) (free t40) (free t58) (free t65) (free t73) (free t74) (free t82) (free t85) (free t88) (free t9) (link h1 h70) (link h1 h82) (link h28 h75) (link h28 h97) (link h3 h79) (link h3 h92) (link h70 h1) (link h70 h75) (link h75 h28) (link h75 h70) (link h79 h3) (link h79 h88) (link h82 h1) (link h82 h92) (link h88 h79) (link h88 h97) (link h92 h3) (link h92 h82) (link h97 h28) (link h97 h88) (open x22) (open x27) (open x31) (open x5) (ready a26) (ready a37) (ready a6) (ready a7))
  (:goal (and (done x22) (done x27) (done x31) (done x5))))
