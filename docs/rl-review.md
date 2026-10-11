# RL review and completion status

This review starts from **dev `4190ea6`**, using the companion mymyr-notes roadmap and the local mifrost sources.
The resulting changes live on **`codex-rl-encoding`**, not dev or main. The independent `r1-encodings` worktree was
not modified or incorporated; its in-progress storage changes need their own review and integration.

## Verdict

RL is meaningful infrastructure, not a placeholder: canonical labelled successor expansion, asynchronous CPU
environments, optional Torch/TorchRL and functional JAX adapters, and replay/search interoperability all use the
native planning dynamics. It should remain a dynamics/observation boundary rather than acquire its own learner
framework or replay implementation. The original defaults and collection contracts, however, made a conventional
learning loop unnecessarily easy to misuse.

## Addressed findings

- Default rewards now favour goal reaching: step 0, goal 1, failure 0. The previous -1 step / zero goal reward
  preferred a one-step irreversible failure over a two-step success. Unit-cost shaping remains explicit.
- Torch observations and ordinary step results own snapshots by default, with an explicit borrowed fast path.
  Reached states precede autoreset; subsequent policy observations include the new instance and goal masks.
- JAX replay rejects autoreset configurations that omit reached states, instead of substituting reset states.
  Source goal masks propagate automatically, including under JIT.
- Initially solved states and terminal dead ends are absorbing, with zero repeated rewards. CPU pool and TorchRL
  reset statuses reflect the actual goal. mctx preserves terminal embeddings, including sampling metadata.
- A seeded uniform/weighted instance sampler works in NumPy, Torch and JAX, including partial resets and compiled
  loops. TorchRL observes `task_id` but assigns resets through `next_task_id`; observed goal masks likewise no
  longer override a newly selected instance's goals.
- An empty single native State expands consistently with the equivalent one-element sequence.
- A pinned optional-framework CI lane enables real CPU adapter tests and explicitly supplies XLA headers to the
  isolated package build. CUDA compilation remains distinct from runtime validation.
- Native `mymyr.encoding` provides classical facts, signed goal satisfaction and existing action readouts, using
  owned packed arrays and the existing DLPack interface. Action source-instance IDs are checked, including empty
  action rows. No foreign planning registry, mandatory framework or copied mifrost implementation is introduced.

`examples/rl_learning.py` exercises a real NumPy policy update through native encoding. It trains on 3–5-object
chains, visits all training instances, and evaluates against random actions on held-out 8/12-object chains.
This is a reproducible schema/count-scoring smoke test, not evidence of general relational planning competence.

## Remaining work

1. **Complete stable storage work separately.** On the reviewed dev baseline, lazy raw rows still depend on first
   touch order; task content fingerprints do not identify those slot maps, and task pickling does not persist them.
   Use native State values or a compatible frozen encoding for persistence. Native relational observations sort
   semantic tuples, but cannot identify a raw row whose original slot map has been lost.
2. **Control cross-size storage.** TaskTable rows still pad to the largest loaded instance, including inactive
   instances. Numeric I32/F64 storage can differ by instance. Active-width buckets and a homogeneous/typed numeric
   consumer contract remain necessary for demanding train-small/evaluate-large workloads.
3. **Extend encoders deliberately.** The first native encoder is CPU/classical and rejects numeric functions and
   constraints, TaskSuite and GPU inputs. Numeric features, device-native encoding and selected additional mifrost
   capabilities need explicit contracts and acceptance tests; this is not full mifrost parity. See
   [the source assessment and staged design](native-encoding-design.md).
4. **Respect action and goal boundaries.** Fixed padded action limits can exclude solving actions; ragged scoring
   avoids that completeness limit but still enumerates successors. HER sees reached states, not source states;
   remove transitions after first relabelled-goal achievement to match absorbing dynamics. Action costs are not
   automatically reflected in sparse rewards. Matching source-instance IDs does not prove that supplied raw rows
   are the original action expansion's source states.
5. **Measure a real consumer.** The smoke example is intentionally tiny. A relational model, numeric-domain case,
   full-loop profiling and actual GPU rollout/replay/capture tests are the next acceptance steps. mctx's root goal
   query currently uses a discarded step; a narrow JAX goal-query FFI would avoid its extra expansion. The general
   CUDA terminal path adds a host goal test; its code compiles, but GPU behaviour is not locally validated.

Final combined validation uses a fresh GCC 16.1 CPU build and the installed Python 3.14t framework environment:
292 selected Python tests pass with warnings treated as errors; 71 CUDA tests/examples or CUDA-stub checks skip.
All 39 selected native RL/encoding tests pass, as do installed frontend and core-only C++ SDK consumers, generated
stub checks, runnable documentation and examples. The fixed-seed smoke example reports 100% learned greedy success
versus 17.97% random success on the held-out tasks. Optional CUDA runtime cases cannot run locally because the
driver is unavailable. This is a targeted development review, not a complete release, sanitizer or GPU certification.
