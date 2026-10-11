# Native encoding: mifrost assessment and implementation boundary

Assessment on 2026-10-11 uses mifrost `dev` at `91b152c21819678a7818865b230774a3b6e800e3`, mymyr `dev` baseline
`4190ea6`, and sibling mymyr-notes. The original analysis was read-only. The implementation now delivered is an
original mymyr-native CPU flat slice, described in [encoding.md](encoding.md), with tests and a small semantic fixture
exported from the actual local mifrost neutral engine. It does not copy mifrost source, claim full family/layout
parity, or implement numeric/CUDA encoding.
The fixture records the inspected checkout separately from the existing extension: its binary build commit was not
independently verified. `scripts/export_mifrost_encoding_fixture.py` reproduces the selected tuple projection with an
installed mifrost or an explicitly supplied native module and matching interpreter.

## Decision

Bring encoding closer through public `mymyr.encoding` and a separate `mymyr::encoding` target. Keep the planning core,
state/atom representation and action labels authoritative. A native encoder consumes these directly and owns only
encoding schema/context, graph-working scratch and completed arrays. It must not reconstruct Pymimir objects or
perform per-state Python atom-name loops.

This revisits mymyr-notes implementation-plan D20 (encoding lives in mifrost, the user's 2026-09-28 decision) after
the new explicit native-module request. D8 (framework-owned replay), removal of symmetry, and the separate exclusion
of WL features and PyG/jraph exports remain in force. Native relational tensors do not require reintroducing them.

## What mifrost can contribute

Mifrost already has a planner-neutral C++ library, `mifrost::neutral_core`, separate from Pymimir/PyTyr adapters.
Canonical algorithms use operation-bearing, statically dispatched Views for atoms, literals, states and actions.
The engine schema is domain-scoped; object/static-fact/default-goal contexts are problem-scoped. This is a suitable
future reuse boundary: a direct borrowed mymyr View adapter, not a complete owning semantic-record mirror per state.

| Capability | Existing implementation | Native mymyr path |
|---|---|---|
| Flat state/goals/actions/history | Neutral C++ flat engine, relation catalog and composition components | First slice provides facts/signed satisfaction/action tuples; native history/subgoal components can follow |
| HGraph and integer Color | Neutral C++ engines | Add only when a consumer needs their array contract; Color IDs are not WL refinement |
| Successor/full/effect transitions | Neutral successor engine; flat transitions reuse horizon substrate | Bridge mymyr CSR expansions and durable labels; preserve target mapping |
| Horizon/lookahead DAGs | Owned neutral SemanticTransitionDAG and horizon engines | Adapt search outputs to an owned encoding DAG only when requested |
| Batches/streams/readout metadata | BatchBuilder/BatchEncoding, stream caches and typed graph fields | Reuse the semantic rules; integrate mymyr-owned arrays, lifetime and persistence |
| Star/object/line/incidence/tuple/bias projections | Neutral derived engine plus graph-specific Python facades | Later array-only families; no mandatory PyG carrier |
| ILG/lifted/object-feature/custom encoders | Python implementations and backend snapshot readers | Selective native ports or a narrow custom array writer, not ready C++ drop-ins |
| Sparse atom composition | Pure Python/PyTorch backend-free core and StateView facade | Useful independent future family; requires an actual native port |
| Numeric observations | No numeric state/function/constraint lane in the neutral Views/DTO | New versioned contract and implementation required |
| CUDA encoding | Existing native DLPack exporter marks CPU buffers | Separate CUDA kernels required, not inherited from embedding mifrost |

The neutral core is approximately 24,700 C++ source/header lines; public Python approximately 23,700 lines. A wholesale
import would be a large second subsystem. The first slice deliberately avoids copying it or acquiring its dependencies.

## Reuse and dependency boundary

Candidate shared components are relation identities/catalogs, goal derivation, canonical flat/HGraph algorithms,
semantic assembly/component rules, schema validation, batch offsets and readout metadata. View-based emission still
allocates compact graph-working records; it is not an allocation-free claim.

Do not import public mifrost Python facades wholesale: backend resolvers recognize Pymimir/PyTyr, several facades
and custom encoders import Torch/PyG, and graph subclasses encode PyG-specific batching conventions. Mymyr's facade
instead accepts native values or array batches plus task IDs, with optional existing DLPack framework export.

Mifrost neutral C++ links Boost headers, fmt, Abseil, parallel-hashmap, unordered_dense, strong_type and range-v3.
Mymyr has Boost/fmt/Abseil for its frontend but not all the additional libraries. If future code reuse is chosen,
pin them in the existing dependency superbuild and use one compiler/dependency prefix. Keep these dependencies out
of the planning core's public interface. Do not run the full mifrost Conan/bootstrap project inside mymyr.

The embedded root build also needs adaptation: Pymimir defaults on, staging/RPATH/wheel behavior is separate, and
C++23 is set globally. Use a small target-based integration layer instead of allowing it to alter mymyr's C++26 build.

Mifrost's neutral Python module joins `NB_DOMAIN pymimir_abi_domain`; Pymimir requires matched nanobind generations,
while PyTyr uses an isolated adapter/capsule boundary. Mymyr uses `NB_STATIC`, no shared domain and a versioned
external C API. Bind native encoding inside mymyr's `_core`, as the first slice does. An independently installed
mifrost BatchEncoding is not automatically the same C++/Python type; interchange through arrays/versioned schema.

## Contracts to preserve

- Stable domain predicate/action IDs; explicit per-instance object IDs and batch offsets; atom slots are storage,
  not model identities. Lazy slot order must not influence tuples or metadata. Distinct normalized action schemas
  and introduced parameters must survive.
- Problem-local derived predicates are scoped, not silently merged because their generated names coincide.
  Domain training/evaluation vocabularies must stay stable; cross-domain TaskSuite encoding needs grouping and
  explicit domain/scatter metadata rather than accidental shared IDs.
- Current atom metadata must decode newly published lazy slots. The first slice reads the task's append-only index
  directly. R1-C remains the owner of the row/slot representation, portability and per-instance width changes.
- Axioms use an independent WorkspaceLease. Negative/static/derived goal satisfaction and final pre-autoreset goals
  must describe the reached observation, not whichever episode reset happens next.
- Action readout rows map one-to-one to the caller's expansion and durable labels; graph-local tuple/node indices
  never become planning identities. Do not enumerate successors merely to observe a state or silently truncate labels.
- Output schemas and buffers have explicit version/dtypes/ownership. Stream appends snapshot completed outputs,
  not borrowed mutable environment rows. Capacities and overflow are visible; optional debug names are not hot-path IDs.

The delivered first slice uses graph-major CSR argument and tuple offsets, int32 checked capacities, globalized
object arguments, nullary counts, preserved expansion label columns and original-action-to-tuple mapping. It is
not mifrost's complete entity/readout carrier or binary layout, and the semantic parity fixture tests only the
selected fact and signed-satisfaction lanes.

## Numeric and CUDA work are real extensions

Mifrost StateView exposes only fluent/derived atoms, the semantic DTO has no numeric lane, and graph fields are
float32/int64. Blind reuse can make different fuel levels observationally identical. First version therefore rejects
numeric/function/constraint tasks. A later contract needs ground function identities/arguments, canonical values,
definedness masks, precision policy, numeric goal/comparison structure and transition/effect features. Raw reinterpretation
of `[bits | numeric words]`, especially across I32/F64 tasks, is not a substitute.

Moving CPU-produced encodings to a GPU is not GPU encoding. Start by measuring CPU batch encoding plus explicit
transfer. A first native CUDA flat path would count set bits/predicate lanes/goals, prefix-sum, scatter relation/object/
target columns and evaluate satisfaction on-device. Reuse semantic schemas rather than CPU strings/hash maps.
Require CPU/CUDA parity, deterministic ordering, per-instance bounds, capacity/overflow reporting and stream-aware
DLPack lifetime before JAX FFI/static-shape capture. Any GPU→CPU→GPU fallback must be explicit.

## Native package versus Git submodule

These are independent choices. `mymyr.encoding` is the supported native API/runtime. A Git submodule only pins
source; it does not adapt planner types, remove dependencies or supply numeric/CUDA features.

1. **Original mymyr first slice — delivered.** Narrow flat contract, no copied GPL-only sources, new dependencies,
   Pymimir/PyTyr or model-framework requirement. Existing semantic tests/fixture guide later reuse decisions.
2. **Pinned neutral source snapshot/subtree plus a mymyr CMake wrapper.** Straightforward checkout/sdist/wheel flow
   if actual upstream code reuse is chosen. Keep provenance and update/parity procedures; do not scatter copied code.
3. **Git submodule at external/mifrost.** Reasonable for active shared development, with commit pin, recursive CI clones
   and neutral-only wrapper. Sdists must contain actual source: a gitlink alone is empty in source archives, and PyPI
   builds must not require a network submodule checkout.
4. **External neutral CMake SDK dependency.** Good for a C++ prototype, but adds another native prefix/library/ABI,
   wheel bundling and RPATH obligations. Not the simplest default Python distribution.

Choose one canonical long-term owner before porting many families: upstream shared neutral algorithms, or native
mymyr algorithms with mifrost as a backend/compatibility facade. Avoid two permanent independent copies.

## Packaging, licensing, stages and gates

Keep import/native encoding NumPy-only, optional Torch/JAX lazy, and installed `mymyr::encoding` consumer tests.
Test free-threaded CPython, Linux/macOS, standalone SDK, fresh sdists without Git/network, and wheel dependency absence.
Mifrost metadata says GPL-3.0-only; mymyr says GPL-3.0-or-later. The first slice copies no mifrost source. Before later
copying or relicensing, retain notices/provenance and resolve combined-distribution intent with the owner; matching
license text files alone do not make their SPDX expressions identical. This is a discrepancy note, not legal advice.

Indicative follow-on scope for one engineer familiar with both projects:

- Contract/canonical-source/licensing decisions: 2–4 days; pin schema/nullary/local-predicate and source boundaries.
- CPU flat hardening and useful learner integration: 1–3 weeks depending on throughput/persistence/collector goals.
- Selected additional native families: 1–2 weeks each according to actual consumer contracts, not nominal parity.
- Numeric and CUDA: independently scoped, likely multiple engineering weeks each. Do not promise them as part of
  a completed Boolean CPU port.

Tests include independent semantic reconstruction, the actual-neutral-engine fixture and reproducible exporter,
positive/negative/static/derived/nullary goals, task/goal identity, lazy growth/order invariance, mixed instance sizes,
problem-local predicate isolation, normalization-introduced parameters, one-to-one action readouts, retained immutable
buffers, reentrant encoding, optional framework export, generated typed stubs and installed
SDK consumers. Future gates add persistence/version mismatch, broader upstream semantic parity, cross-domain grouping,
TSan, numeric distinguishability/undefinedness, non-default CUDA streams and JIT/capture capacity constraints.

Source landmarks, relative to the sibling-repository workspace: `mifrost/src/CMakeLists.txt:587`,
`mifrost/src/_core/mifrost/core/views/concepts.hpp:275`, `mifrost/src/_core/mifrost/core/semantic/records.hpp:177`,
`mifrost/src/_core/mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp:129`,
`mifrost/src/_core/mifrost/core/graph_fields.hpp:14`, `mifrost/src/_core/mifrost/core/dlpack_utils.hpp:81`,
`mifrost/src/mifrost/encoders/sparse_atom.py:1`, `mifrost/pyproject.toml:30`,
`mymyr-notes/analysis/implementation-plan.md:39`, `mymyr-notes/docs/ROADMAP.md:614`.
