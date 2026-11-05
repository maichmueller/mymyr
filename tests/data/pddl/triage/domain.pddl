; Copies of a gadget chain: a copy is done when a walk links one cell per gadget up to a terminal gadget (linking a
; cell needs a road to every linked cell of the gadgets it borders: a universal precondition with an implication) or
; when an alternative walk patches the chain and seals it.
(define (domain cfi-bench-triage)
 (:requirements :strips :typing :universal-preconditions :disjunctive-preconditions)
 (:types copy gadget cell)
 (:predicates
   (in ?c - cell ?g - gadget)
   (road ?c - cell ?d - cell)
   (of ?g - gadget ?k - copy)
   (first ?g - gadget)
   (next ?g - gadget ?h - gadget)
   (terminal ?g - gadget)
   (anchor ?g - gadget)
   (border ?g - gadget ?u - gadget)
   (fresh ?k - copy)
   (cur ?g - gadget)
   (linked ?c - cell)
   (done ?k - copy)
   (pcur ?g - gadget)
   (sealed ?k - copy))
 (:action begin
   :parameters (?k - copy ?g - gadget)
   :precondition (and (fresh ?k) (of ?g ?k) (first ?g))
   :effect (and (not (fresh ?k)) (cur ?g)))
 (:action link
   :parameters (?g - gadget ?h - gadget ?c - cell)
   :precondition (and (cur ?g) (next ?g ?h) (in ?c ?g)
     (forall (?u - gadget ?p - cell) (imply (and (border ?g ?u) (in ?p ?u) (linked ?p)) (road ?c ?p))))
   :effect (and (not (cur ?g)) (cur ?h) (linked ?c)))
 (:action close
   :parameters (?k - copy ?g - gadget)
   :precondition (and (cur ?g) (terminal ?g) (of ?g ?k))
   :effect (and (not (cur ?g)) (done ?k)))
 (:action alt-begin
   :parameters (?k - copy ?g - gadget)
   :precondition (and (fresh ?k) (of ?g ?k) (first ?g))
   :effect (and (not (fresh ?k)) (pcur ?g)))
 (:action patch
   :parameters (?g - gadget ?h - gadget)
   :precondition (and (pcur ?g) (next ?g ?h))
   :effect (and (not (pcur ?g)) (pcur ?h)))
 (:action seal
   :parameters (?k - copy ?g - gadget)
   :precondition (and (pcur ?g) (terminal ?g) (of ?g ?k))
   :effect (and (not (pcur ?g)) (sealed ?k)))
 (:action alt-close
   :parameters (?k - copy)
   :precondition (sealed ?k)
   :effect (and (not (sealed ?k)) (done ?k))))
