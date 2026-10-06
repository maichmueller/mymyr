"""Philosophers problems (IPC 2004 promela, domain `protocol`) with N philosophers, in the layout of the fork's
data/philosophers/test_problem.pddl (N = 2). Usage: python3 gen_phil.py N > phil-N.pddl"""
import sys

n = int(sys.argv[1])
out = []
w = out.append
w("(define (problem phil-%d)" % n)
w("(:domain protocol)")
w("(:objects")
for i in range(n):
    w("  philosopher-%d" % i)
w("  - process")
for i in range(n):
    w("  forks-%d-" % i)
w("  - queue")
w("  queue-1 - queuetype")
w("  qs-0 - queue-state")
w("  empty fork - message")
w("  zero one - number_")
w("  philosopher - proctype")
w("  state-1 state-6 state-3 state-4 state-5 - state")
w("  forks--pid-Wfork forks--pid-Rfork forks-__-pidp1__N_-Rfork forks-__-pidp1__N_-Wfork - transition")
w(")")
w("(:init")
w("  (queue-next queue-1 qs-0 qs-0)")
w("  (is-not-max queue-1 zero)")
w("  (is-max queue-1 one)")
for i in range(n):
    w("  (pending philosopher-%d)" % i)
    w("  (at-process philosopher-%d state-1)" % i)
    w("  (is-a-process philosopher-%d philosopher)" % i)
w("  (is-zero zero)")
w("  (dec one zero)")
w("  (inc zero one)")
w("  (is-not-zero one)")
for i in range(n):
    w("  (is-a-queue forks-%d- queue-1)" % i)
    w("  (queue-head forks-%d- qs-0)" % i)
    w("  (queue-tail forks-%d- qs-0)" % i)
    w("  (queue-head-msg forks-%d- empty)" % i)
    w("  (queue-size forks-%d- zero)" % i)
    w("  (settled forks-%d-)" % i)
for t in ("forks--pid-Wfork", "forks--pid-Rfork", "forks-__-pidp1__N_-Rfork", "forks-__-pidp1__N_-Wfork"):
    w("  (trans-msg %s fork)" % t)
for i in range(n):
    j = (i + 1) % n
    w("  (writes philosopher-%d forks-%d- forks--pid-Wfork)" % (i, i))
    w("  (reads philosopher-%d forks-%d- forks--pid-Rfork)" % (i, i))
    w("  (reads philosopher-%d forks-%d- forks-__-pidp1__N_-Rfork)" % (i, j))
    w("  (writes philosopher-%d forks-%d- forks-__-pidp1__N_-Wfork)" % (i, j))
w("  (trans philosopher forks--pid-Wfork state-1 state-6)")
w("  (trans philosopher forks--pid-Rfork state-6 state-3)")
w("  (trans philosopher forks-__-pidp1__N_-Rfork state-3 state-4)")
w("  (trans philosopher forks--pid-Wfork state-4 state-5)")
w("  (trans philosopher forks-__-pidp1__N_-Wfork state-5 state-6)")
w(")")
w("(:goal (and")
for i in range(n):
    w("  (blocked philosopher-%d)" % i)
w("))")
w(")")
print("\n".join(out))
