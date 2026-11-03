; agents = 4, jobs = 2, hubs = 10
(define (problem p055) (:domain binding-hubs)
  (:objects a5 a7 a21 a39 - agent
            t13 t28 t34 t37 t40 t46 t47 t53 t55 t68 t78 t86 t89 t90 t96 t111 t140 t175 t190 t191 - tool
            h2 h3 h7 h13 h27 h42 h52 h65 h69 h98 - hub
            x1 x6 - job)
  (:init (end t111 h27) (end t111 h7) (end t13 h3) (end t13 h65) (end t140 h42) (end t140 h65) (end t175 h13) (end t175 h98) (end t190 h42) (end t190 h69) (end t191 h2) (end t191 h7) (end t28 h27) (end t28 h52) (end t34 h13) (end t34 h52) (end t37 h2) (end t37 h42) (end t40 h13) (end t40 h3) (end t46 h3) (end t46 h69) (end t47 h27) (end t47 h98) (end t53 h27) (end t53 h42) (end t55 h13) (end t55 h7) (end t68 h2) (end t68 h3) (end t78 h69) (end t78 h7) (end t86 h52) (end t86 h65) (end t89 h2) (end t89 h98) (end t90 h65) (end t90 h98) (end t96 h52) (end t96 h69) (free t111) (free t13) (free t140) (free t175) (free t190) (free t191) (free t28) (free t34) (free t37) (free t40) (free t46) (free t47) (free t53) (free t55) (free t68) (free t78) (free t86) (free t89) (free t90) (free t96) (link h13 h42) (link h13 h7) (link h2 h3) (link h2 h52) (link h27 h3) (link h27 h98) (link h3 h2) (link h3 h27) (link h42 h13) (link h42 h69) (link h52 h2) (link h52 h65) (link h65 h52) (link h65 h7) (link h69 h42) (link h69 h98) (link h7 h13) (link h7 h65) (link h98 h27) (link h98 h69) (open x1) (open x6) (ready a21) (ready a39) (ready a5) (ready a7))
  (:goal (and (done x1) (done x6))))
