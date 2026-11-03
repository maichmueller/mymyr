; Binding, hubs variant: a control for context marks. A tool spans two hubs
; (end), the hubs form a cycle (link), and a job needs a tool whose two hubs
; are linked: a valid tool closes a triangle tool - hub - hub. pick takes any
; agent and any tool and names no hub, so whether a pick is right is a property
; of objects the action does not contain; work names the hubs (see generate.py
; --hubs).
(define (domain binding-hubs)
  (:requirements :strips :typing)
  (:types agent tool hub job)
  (:predicates
    (link ?h - hub ?g - hub)
    (end ?t - tool ?h - hub)
    (free ?t - tool)
    (ready ?a - agent)
    (holding ?a - agent ?t - tool)
    (open ?x - job)
    (done ?x - job))
  (:action pick
    :parameters (?a - agent ?t - tool)
    :precondition (and (ready ?a) (free ?t))
    :effect (and (holding ?a ?t) (not (ready ?a)) (not (free ?t))))
  (:action drop
    :parameters (?a - agent ?t - tool)
    :precondition (holding ?a ?t)
    :effect (and (ready ?a) (free ?t) (not (holding ?a ?t))))
  (:action work
    :parameters (?a - agent ?t - tool ?h - hub ?g - hub ?x - job)
    :precondition (and (holding ?a ?t) (end ?t ?h) (end ?t ?g) (link ?h ?g) (open ?x))
    :effect (and (done ?x) (free ?t) (not (open ?x)) (not (holding ?a ?t)))))
