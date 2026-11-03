; agents = 3, jobs = 1, hubs = 8
(define (problem p022) (:domain binding-hubs)
  (:objects a5 a16 a24 - agent
            t20 t31 t39 t40 t55 t67 t75 t84 t95 t100 t109 t114 t115 t141 t153 t160 - tool
            h11 h13 h24 h42 h48 h55 h60 h75 - hub
            x9 - job)
  (:init (end t100 h11) (end t100 h13) (end t109 h48) (end t109 h60) (end t114 h13) (end t114 h55) (end t115 h13) (end t115 h60) (end t141 h11) (end t141 h75) (end t153 h13) (end t153 h24) (end t160 h11) (end t160 h42) (end t20 h24) (end t20 h55) (end t31 h48) (end t31 h75) (end t39 h24) (end t39 h60) (end t40 h11) (end t40 h48) (end t55 h24) (end t55 h75) (end t67 h42) (end t67 h75) (end t75 h42) (end t75 h48) (end t84 h55) (end t84 h60) (end t95 h42) (end t95 h55) (free t100) (free t109) (free t114) (free t115) (free t141) (free t153) (free t160) (free t20) (free t31) (free t39) (free t40) (free t55) (free t67) (free t75) (free t84) (free t95) (link h11 h24) (link h11 h42) (link h13 h60) (link h13 h75) (link h24 h11) (link h24 h55) (link h42 h11) (link h42 h60) (link h48 h55) (link h48 h75) (link h55 h24) (link h55 h48) (link h60 h13) (link h60 h42) (link h75 h13) (link h75 h48) (open x9) (ready a16) (ready a24) (ready a5))
  (:goal (and (done x9))))
