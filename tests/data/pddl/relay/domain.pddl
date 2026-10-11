(define (domain relay)
 (:requirements :strips :typing :negative-preconditions :existential-preconditions)
 (:types node)
 (:predicates (at ?n - node) (link ?a ?b - node) (lit ?n - node) (visited ?n - node))
 (:action step
  :parameters (?from ?to - node)
  :precondition (and (at ?from) (link ?from ?to)
                     (exists (?w - node) (and (lit ?w) (link ?w ?to))))
  :effect (and (not (at ?from)) (at ?to) (visited ?to)))
 (:action light
  :parameters (?n - node)
  :precondition (and (at ?n) (not (lit ?n)))
  :effect (lit ?n)))
