(define (domain blocks-flags)
 (:requirements :strips)
 (:predicates (clear ?x) (ontable ?x) (on ?x ?y) (holding ?x) (handempty) (moved) (stacked))
 (:action pick-up
  :parameters (?x)
  :precondition (and (clear ?x) (ontable ?x) (handempty))
  :effect (and (not (ontable ?x)) (not (clear ?x)) (not (handempty)) (holding ?x) (moved)))
 (:action put-down
  :parameters (?x)
  :precondition (holding ?x)
  :effect (and (not (holding ?x)) (clear ?x) (handempty) (ontable ?x)))
 (:action stack
  :parameters (?x ?y)
  :precondition (and (holding ?x) (clear ?y))
  :effect (and (not (holding ?x)) (not (clear ?y)) (clear ?x) (handempty) (on ?x ?y) (stacked)))
 (:action unstack
  :parameters (?x ?y)
  :precondition (and (on ?x ?y) (clear ?x) (handempty))
  :effect (and (holding ?x) (clear ?y) (not (clear ?x)) (not (handempty)) (not (on ?x ?y)) (moved) (not (stacked)))))
