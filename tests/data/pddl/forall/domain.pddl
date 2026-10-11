(define (domain wiring)
  (:requirements :adl :typing)
  (:types room switch - object)
  (:predicates (lit ?r - room) (wired ?s - switch ?r - room) (on ?s - switch))

  (:action flip
    :parameters (?s - switch)
    :precondition (not (on ?s))
    :effect (and (on ?s)
                 (forall (?r - room) (when (wired ?s ?r) (lit ?r)))))

  (:action confirm
    :parameters (?s - switch)
    :precondition (forall (?r - room) (imply (wired ?s ?r) (lit ?r)))
    :effect (on ?s)))
