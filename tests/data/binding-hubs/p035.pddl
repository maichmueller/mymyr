; agents = 3, jobs = 3, hubs = 10
(define (problem p035) (:domain binding-hubs)
  (:objects a14 a20 a29 - agent
            t7 t14 t30 t31 t73 t76 t83 t85 t105 t108 t111 t112 t119 t125 t134 t135 t145 t177 t193 t195 - tool
            h40 h42 h49 h56 h76 h77 h79 h93 h94 h97 - hub
            x3 x8 x29 - job)
  (:init (end t105 h42) (end t105 h97) (end t108 h77) (end t108 h93) (end t111 h56) (end t111 h76) (end t112 h42) (end t112 h94) (end t119 h56) (end t119 h77) (end t125 h49) (end t125 h79) (end t134 h40) (end t134 h79) (end t135 h42) (end t135 h76) (end t14 h40) (end t14 h94) (end t145 h56) (end t145 h94) (end t177 h79) (end t177 h93) (end t193 h40) (end t193 h76) (end t195 h49) (end t195 h77) (end t30 h40) (end t30 h77) (end t31 h49) (end t31 h97) (end t7 h42) (end t7 h79) (end t73 h93) (end t73 h94) (end t76 h49) (end t76 h76) (end t83 h56) (end t83 h97) (end t85 h93) (end t85 h97) (free t105) (free t108) (free t111) (free t112) (free t119) (free t125) (free t134) (free t135) (free t14) (free t145) (free t177) (free t193) (free t195) (free t30) (free t31) (free t7) (free t73) (free t76) (free t83) (free t85) (link h40 h76) (link h40 h97) (link h42 h77) (link h42 h97) (link h49 h77) (link h49 h94) (link h56 h79) (link h56 h94) (link h76 h40) (link h76 h93) (link h77 h42) (link h77 h49) (link h79 h56) (link h79 h93) (link h93 h76) (link h93 h79) (link h94 h49) (link h94 h56) (link h97 h40) (link h97 h42) (open x29) (open x3) (open x8) (ready a14) (ready a20) (ready a29))
  (:goal (and (done x29) (done x3) (done x8))))
