; Beacons on a directed network: a lit beacon is powered when it is a source or a powered beacon links to it (a
; recursive derived predicate); powered beacons cannot be doused, and a powered beacon can block an unlit neighbour.
(define (domain beacons)
  (:requirements :strips :typing :negative-preconditions :existential-preconditions :derived-predicates)
  (:types node)
  (:predicates (edge ?a - node ?b - node) (source ?n - node) (lit ?n - node) (blocked ?n - node) (powered ?n - node))
  (:derived (powered ?n - node) (and (lit ?n) (source ?n)))
  (:derived (powered ?n - node) (exists (?m - node) (and (lit ?n) (edge ?m ?n) (powered ?m))))
  (:action light
    :parameters (?n - node)
    :precondition (and (not (lit ?n)) (not (blocked ?n)))
    :effect (lit ?n))
  (:action douse
    :parameters (?n - node)
    :precondition (and (lit ?n) (not (powered ?n)))
    :effect (not (lit ?n)))
  (:action block
    :parameters (?m - node ?n - node)
    :precondition (and (powered ?m) (edge ?m ?n) (not (lit ?n)) (not (blocked ?n)))
    :effect (blocked ?n)))
