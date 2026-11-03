#pragma once
// Evaluators written in Python (mymyr._typing.Evaluator): the heuristics of the best-first searches and the scores of
// a scored layer ordering. An evaluator is a callable state -> float or an object with evaluate(state) and/or
// evaluate_batch(batch); a batch is a mymyr.search.StateBatch, the packed state rows of the states (and, when the
// object declares an `encoder`, their encoding), never a list of State objects. A SharedEvaluator wraps an evaluator
// for many searches at once: their batches are coalesced into one evaluate_batch call.

#include "py_table.hpp"
#include "py_task.hpp"

#include "mymyr/core/types.hpp"
#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/heuristics/shared.hpp"
#include "mymyr/search/control.hpp"
#include "mymyr/state/state.hpp"

#include <nanobind/nanobind.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace mymyr::python
{
namespace nb = nanobind;

/// Collects the first exception of any callback, and cancels the search when one occurs.
struct CallbackErrors
{
    search::CancelToken cancel;
    std::mutex m;
    std::unique_ptr<nb::python_error> first;  // touched with an attached thread state only
    std::atomic<bool> failed{false};

    /// Call with an attached thread state, inside the catch of a python_error.
    void record(nb::python_error&& e)
    {
        {
            std::lock_guard lock(m);
            if (!first)
                first = std::make_unique<nb::python_error>(std::move(e));
        }
        failed.store(true, std::memory_order_relaxed);
        cancel.request();
    }

    /// Call with an attached thread state after the search: re-raises the first error.
    void rethrow()
    {
        if (first)
        {
            std::unique_ptr<nb::python_error> e = std::move(first);
            throw nb::python_error(std::move(*e));
        }
    }
};

/// What the values of an evaluator mean: heuristic values (>= 0, +inf a dead end) or scores (any number but NaN).
enum class ValueKind : u8
{
    Heuristic,
    Score,
};

/// An evaluator object, parsed once: its methods and its encoder. Read-only after construction; the calls need an
/// attached thread state.
class Evaluator
{
public:
    /// Parses `obj` (with an attached thread state); TypeError when it is no evaluator.
    explicit Evaluator(nb::handle obj);
    Evaluator(const Evaluator&) = delete;
    Evaluator& operator=(const Evaluator&) = delete;
    ~Evaluator();

    nb::object obj, one, batch, preferred;  // the object and its evaluate, evaluate_batch and preferred_actions
    nb::object encoder, encode_options;     // the declared encoder and the keywords of its encode call (a dict)
    TableRef table;                         // the encoder's table (encoder.table)

    [[nodiscard]] bool has_encoder() const noexcept { return encoder.is_valid(); }
    /// The instance of the encoder's table whose rows the states of `o`'s task are (-1 without an encoder): the
    /// task itself, or an instance of the same fingerprint and atom encoding. ValueError when there is none.
    [[nodiscard]] i32 instance_of(const Owner& o) const;
};

/// One search's share of a batch: its states, its task, the instance of the encoder's table, where its values go.
struct BatchPart
{
    std::span<const StateView> states;
    const Owner* owner = nullptr;
    i32 instance = -1;
    std::span<f64> out;
    ValueKind kind = ValueKind::Heuristic;
};

/// Calls evaluator.evaluate_batch once with the states of all parts as one StateBatch (encoded first when the
/// evaluator declares an encoder) and writes each part's values into its `out`. Needs an attached thread state;
/// raises nb::python_error for an exception of the evaluator or a value that is not a value of the part's kind.
void evaluate_parts(const std::shared_ptr<const Evaluator>& e, std::span<BatchPart> parts);

/// The float of a Python number as a value of `kind` (python_error for non-numbers and values outside the kind).
[[nodiscard]] f64 value_of(PyObject* x, ValueKind kind);

/// mymyr.search.SharedEvaluator: an evaluator shared by searches on any number of threads, coalescing their batches
/// into one evaluate_batch call (heuristics::SharedEvaluator; each request's tag is its BatchPart).
class PySharedEvaluator
{
public:
    PySharedEvaluator(nb::handle evaluator, u64 max_batch, double max_wait);
    PySharedEvaluator(const PySharedEvaluator&) = delete;
    PySharedEvaluator& operator=(const PySharedEvaluator&) = delete;

    std::shared_ptr<const Evaluator> evaluator;
    std::unique_ptr<heuristics::SharedEvaluator> core;
};

/// A search's view of a Python evaluator (heuristics::Kind::Custom): an evaluator object of its own or a
/// SharedEvaluator. Every call attaches the thread state. The first exception is recorded in `errors`, which cancels
/// the search and re-raises it afterwards; evaluations after it return +inf without calling Python.
class PyEvaluatorHeuristic final : public heuristics::Heuristic
{
public:
    /// With an attached thread state. `kind`: Score for a layer ordering's scores.
    PyEvaluatorHeuristic(nb::handle obj, Owner owner, CallbackErrors& errors, ValueKind kind = ValueKind::Heuristic);
    PyEvaluatorHeuristic(const PyEvaluatorHeuristic&) = delete;
    PyEvaluatorHeuristic& operator=(const PyEvaluatorHeuristic&) = delete;
    ~PyEvaluatorHeuristic() override;

    [[nodiscard]] heuristics::Kind kind() const noexcept override { return heuristics::Kind::Custom; }
    heuristics::Value evaluate(StateView s) override;
    /// The goals of GoalSpec::AnyOf are not passed on: a Python heuristic estimates whatever it estimates.
    heuristics::Value evaluate(StateView s, std::span<const search::GoalSpec::AtomGoal> /*goals*/) override
    {
        return evaluate(s);
    }
    void evaluate_batch(std::span<const StateView> states, std::span<heuristics::Value> out) override;
    [[nodiscard]] bool batched() const noexcept override { return m_batched; }
    [[nodiscard]] bool provides_preferred() const noexcept override { return m_eval && m_eval->preferred.is_valid(); }
    [[nodiscard]] bool preferred(const ActionLabel& a) const override;

private:
    void count(heuristics::Value v);
    void load_preferred(nb::handle actions);
    /// Records the exception of a shared call as this search's error (thread state attached).
    void record(const std::exception_ptr& error);

    Owner m_owner;
    CallbackErrors& m_errors;
    ValueKind m_kind;
    std::shared_ptr<const Evaluator> m_eval;  // the evaluator (a SharedEvaluator's own)
    nb::object m_shared_obj;                  // the SharedEvaluator object, or none
    PySharedEvaluator* m_shared = nullptr;
    i32 m_instance = -1;
    bool m_batched = false;
    std::vector<Action> m_pref;  // sorted
};

/// The scorer of LayerOrdering::Kind::Scored from an evaluator object or a SharedEvaluator: called on the searching
/// thread with the thread state detached; values are scores (lower first). Errors go to `errors` (the scores are then
/// 0, and the search is cancelled).
[[nodiscard]] std::function<void(std::span<const StateView>, std::span<f64>)> layer_scorer(nb::handle obj, const Owner& owner,
                                                                                           CallbackErrors& errors);

/// Whether `obj` is an evaluator for a layer ordering (rather than the name of a kind).
[[nodiscard]] bool is_evaluator(nb::handle obj);

/// Registers StateBatch and SharedEvaluator in the search module.
void bind_evaluators(nb::module_& m);
}  // namespace mymyr::python
