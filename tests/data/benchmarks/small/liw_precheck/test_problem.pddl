(define (problem liw-precheck-p1)
  (:domain liw-precheck)
  ;; `q` is true from the start on purpose: `flip-landmark` then adds nothing but the landmark, so
  ;; the only pair it can expose is `(lm1, {q})` -- a pair whose free half is an atom the action
  ;; never touches.
  (:init (p) (q))
  (:goal (and (lm1) (lm2) (done))))
