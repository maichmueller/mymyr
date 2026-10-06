;; Philosophers with 5 philosophers (712,192 states), from gen_phil.py 5;
;; domain: the fork's data/philosophers/domain.pddl
(define (problem phil-5)
(:domain protocol)
(:objects
  philosopher-0
  philosopher-1
  philosopher-2
  philosopher-3
  philosopher-4
  - process
  forks-0-
  forks-1-
  forks-2-
  forks-3-
  forks-4-
  - queue
  queue-1 - queuetype
  qs-0 - queue-state
  empty fork - message
  zero one - number_
  philosopher - proctype
  state-1 state-6 state-3 state-4 state-5 - state
  forks--pid-Wfork forks--pid-Rfork forks-__-pidp1__N_-Rfork forks-__-pidp1__N_-Wfork - transition
)
(:init
  (queue-next queue-1 qs-0 qs-0)
  (is-not-max queue-1 zero)
  (is-max queue-1 one)
  (pending philosopher-0)
  (at-process philosopher-0 state-1)
  (is-a-process philosopher-0 philosopher)
  (pending philosopher-1)
  (at-process philosopher-1 state-1)
  (is-a-process philosopher-1 philosopher)
  (pending philosopher-2)
  (at-process philosopher-2 state-1)
  (is-a-process philosopher-2 philosopher)
  (pending philosopher-3)
  (at-process philosopher-3 state-1)
  (is-a-process philosopher-3 philosopher)
  (pending philosopher-4)
  (at-process philosopher-4 state-1)
  (is-a-process philosopher-4 philosopher)
  (is-zero zero)
  (dec one zero)
  (inc zero one)
  (is-not-zero one)
  (is-a-queue forks-0- queue-1)
  (queue-head forks-0- qs-0)
  (queue-tail forks-0- qs-0)
  (queue-head-msg forks-0- empty)
  (queue-size forks-0- zero)
  (settled forks-0-)
  (is-a-queue forks-1- queue-1)
  (queue-head forks-1- qs-0)
  (queue-tail forks-1- qs-0)
  (queue-head-msg forks-1- empty)
  (queue-size forks-1- zero)
  (settled forks-1-)
  (is-a-queue forks-2- queue-1)
  (queue-head forks-2- qs-0)
  (queue-tail forks-2- qs-0)
  (queue-head-msg forks-2- empty)
  (queue-size forks-2- zero)
  (settled forks-2-)
  (is-a-queue forks-3- queue-1)
  (queue-head forks-3- qs-0)
  (queue-tail forks-3- qs-0)
  (queue-head-msg forks-3- empty)
  (queue-size forks-3- zero)
  (settled forks-3-)
  (is-a-queue forks-4- queue-1)
  (queue-head forks-4- qs-0)
  (queue-tail forks-4- qs-0)
  (queue-head-msg forks-4- empty)
  (queue-size forks-4- zero)
  (settled forks-4-)
  (trans-msg forks--pid-Wfork fork)
  (trans-msg forks--pid-Rfork fork)
  (trans-msg forks-__-pidp1__N_-Rfork fork)
  (trans-msg forks-__-pidp1__N_-Wfork fork)
  (writes philosopher-0 forks-0- forks--pid-Wfork)
  (reads philosopher-0 forks-0- forks--pid-Rfork)
  (reads philosopher-0 forks-1- forks-__-pidp1__N_-Rfork)
  (writes philosopher-0 forks-1- forks-__-pidp1__N_-Wfork)
  (writes philosopher-1 forks-1- forks--pid-Wfork)
  (reads philosopher-1 forks-1- forks--pid-Rfork)
  (reads philosopher-1 forks-2- forks-__-pidp1__N_-Rfork)
  (writes philosopher-1 forks-2- forks-__-pidp1__N_-Wfork)
  (writes philosopher-2 forks-2- forks--pid-Wfork)
  (reads philosopher-2 forks-2- forks--pid-Rfork)
  (reads philosopher-2 forks-3- forks-__-pidp1__N_-Rfork)
  (writes philosopher-2 forks-3- forks-__-pidp1__N_-Wfork)
  (writes philosopher-3 forks-3- forks--pid-Wfork)
  (reads philosopher-3 forks-3- forks--pid-Rfork)
  (reads philosopher-3 forks-4- forks-__-pidp1__N_-Rfork)
  (writes philosopher-3 forks-4- forks-__-pidp1__N_-Wfork)
  (writes philosopher-4 forks-4- forks--pid-Wfork)
  (reads philosopher-4 forks-4- forks--pid-Rfork)
  (reads philosopher-4 forks-0- forks-__-pidp1__N_-Rfork)
  (writes philosopher-4 forks-0- forks-__-pidp1__N_-Wfork)
  (trans philosopher forks--pid-Wfork state-1 state-6)
  (trans philosopher forks--pid-Rfork state-6 state-3)
  (trans philosopher forks-__-pidp1__N_-Rfork state-3 state-4)
  (trans philosopher forks--pid-Wfork state-4 state-5)
  (trans philosopher forks-__-pidp1__N_-Wfork state-5 state-6)
)
(:goal (and
  (blocked philosopher-0)
  (blocked philosopher-1)
  (blocked philosopher-2)
  (blocked philosopher-3)
  (blocked philosopher-4)
))
)
