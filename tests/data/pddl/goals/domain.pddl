;;; Goals of every shape over one small typed domain: the problems differ only in their goal.
;;; Objects of the problems: places a, b and the constant home; things t1, t2, big b1 and the constant k0.

(define (domain goals)
  (:requirements :strips :typing :negative-preconditions :disjunctive-preconditions :equality
                 :existential-preconditions :universal-preconditions :derived-predicates)
  (:types place thing - object
          big ghost - thing)
  (:constants home - place
              k0 - thing)
  (:predicates (lit ?p - place)
               (ok ?t - thing)
               (at ?t - thing ?p - place)
               (road ?p ?q - place)
               (heavy ?t - thing)
               (seen ?x - object)
               (flag)
               (done))

  (:derived (done) (and (flag) (lit home)))

  (:action light
    :parameters (?p - place)
    :precondition (and)
    :effect (lit ?p))
  (:action unlight
    :parameters (?p - place)
    :precondition (and)
    :effect (not (lit ?p)))
  (:action mark
    :parameters (?t - thing)
    :precondition (and)
    :effect (ok ?t))
  (:action unmark
    :parameters (?t - thing)
    :precondition (and)
    :effect (not (ok ?t)))
  (:action put
    :parameters (?t - thing ?p - place)
    :precondition (and)
    :effect (at ?t ?p))
  (:action see
    :parameters (?x - object)
    :precondition (and)
    :effect (seen ?x))
  (:action raise
    :parameters ()
    :precondition (and)
    :effect (flag)))
