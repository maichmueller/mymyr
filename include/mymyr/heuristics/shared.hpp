#pragma once
// One evaluator for many searches: the batches of searches running on any number of threads are coalesced into one
// call (a learned model on an accelerator evaluates hundreds of states per call at the cost of one).
//
//   heuristics::SharedEvaluator shared([&](std::span<const heuristics::SharedEvaluator::Request> requests) {
//       for (const auto& r : requests) model.evaluate(r.states, r.out);   // or one call over all of them
//   });
//   // per search, on its own thread:
//   heuristics::SharedHeuristic h(shared);
//   search::BestFirstOptions o;
//   o.evaluator = &h;
//   o.expansion_batch = 16;
//   auto r = search::gbfs_eager(task, o);
//
// Protocol. A search hands over a request (its states and where their values go) and waits. Whenever no call is
// running and the queue is due, one waiting thread takes requests off the queue, in arrival order and whole (never
// split; a request larger than max_batch is called alone), until they hold max_batch states, and runs the call on
// them; the others wait. The queue is due when it holds max_batch states, when every search using the evaluator
// (between enter() and leave(); SharedHeuristic does both) is waiting, or when its oldest request has waited max_wait.
// So the call runs on one of the searching threads, never on two at once, and no thread exists besides the searches'.
//
// Results. A search's values, and so its result, are those it would get alone whenever the call computes the value of
// a state from that state alone (not from the other states of the call); the coalescing (the number of calls and their
// composition) depends on the timing. A call that throws fails every request in it: evaluate() returns the exception
// to each of their searches. Waiting is bounded by max_wait plus the running call: a search whose deadline passes or
// which is cancelled while it waits notices it after its request is answered.

#include "mymyr/core/types.hpp"
#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/state/state.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <span>

namespace mymyr::heuristics
{
class SharedEvaluator
{
public:
    using Clock = std::chrono::steady_clock;
    /// One search's request: its states, the values to write (the same length), and the caller's tag (passed through).
    struct Request
    {
        std::span<const StateView> states;
        std::span<Value> out;
        const void* tag = nullptr;
    };
    /// Writes the values of every request's states into its `out`; may throw.
    using Call = std::function<void(std::span<const Request> requests)>;

    /// Throws std::invalid_argument for an empty call, max_batch 0 or a negative max_wait.
    explicit SharedEvaluator(Call call, u64 max_batch = 1024, Clock::duration max_wait = std::chrono::milliseconds(1));
    SharedEvaluator(const SharedEvaluator&) = delete;
    SharedEvaluator& operator=(const SharedEvaluator&) = delete;

    /// A search starts (enter) or ends (leave) using the evaluator.
    void enter();
    void leave();
    /// Evaluates the request among the coalesced ones; blocks until it is answered. Returns the exception of the
    /// call that answered it (null on success).
    [[nodiscard]] std::exception_ptr evaluate(const Request& request);

    [[nodiscard]] u64 max_batch() const noexcept { return m_max_batch; }
    [[nodiscard]] Clock::duration max_wait() const noexcept { return m_max_wait; }
    /// Calls made, states evaluated over all calls, the most states of one call.
    [[nodiscard]] u64 calls() const noexcept { return m_calls.load(std::memory_order_relaxed); }
    [[nodiscard]] u64 rows() const noexcept { return m_rows.load(std::memory_order_relaxed); }
    [[nodiscard]] u64 largest() const noexcept { return m_largest.load(std::memory_order_relaxed); }

private:
    struct Waiting
    {
        const Request* request;
        Clock::time_point since;
        bool done = false;
        std::exception_ptr error{};
    };
    [[nodiscard]] bool due(Clock::time_point now) const;
    void lead(std::unique_lock<std::mutex>& lock);

    Call m_call;
    u64 m_max_batch;
    Clock::duration m_max_wait;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::deque<Waiting*> m_queue;
    u64 m_queued_rows = 0;
    u32 m_active = 0;   // searches between enter() and leave()
    u32 m_waiting = 0;  // searches whose request is not answered
    bool m_calling = false;
    std::atomic<u64> m_calls{0}, m_rows{0}, m_largest{0};
};

/// A search's heuristic that evaluates through a SharedEvaluator (Kind::Custom, batched): enter() on construction,
/// leave() on destruction. An exception of the call is rethrown (the search fails with its message).
class SharedHeuristic final : public Heuristic
{
public:
    explicit SharedHeuristic(SharedEvaluator& shared, const void* tag = nullptr);
    SharedHeuristic(const SharedHeuristic&) = delete;
    SharedHeuristic& operator=(const SharedHeuristic&) = delete;
    ~SharedHeuristic() override;

    [[nodiscard]] Kind kind() const noexcept override { return Kind::Custom; }
    Value evaluate(StateView s) override;
    /// The goals are not passed on: the shared evaluator estimates whatever it estimates.
    Value evaluate(StateView s, std::span<const search::GoalSpec::AtomGoal> /*goals*/) override { return evaluate(s); }
    void evaluate_batch(std::span<const StateView> states, std::span<Value> out) override;
    [[nodiscard]] bool batched() const noexcept override { return true; }

private:
    SharedEvaluator& m_shared;
    const void* m_tag;
};
}  // namespace mymyr::heuristics
