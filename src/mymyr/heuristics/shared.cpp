// One evaluator for many searches (heuristics/shared.hpp): the request queue and the leading thread's call.

#include "mymyr/heuristics/shared.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mymyr::heuristics
{
SharedEvaluator::SharedEvaluator(Call call, u64 max_batch, Clock::duration max_wait)
    : m_call(std::move(call)), m_max_batch(max_batch), m_max_wait(max_wait)
{
    if (!m_call)
        throw std::invalid_argument("mymyr SharedEvaluator: the call is empty");
    if (max_batch == 0)
        throw std::invalid_argument("mymyr SharedEvaluator: max_batch must be positive");
    if (max_wait < Clock::duration::zero())
        throw std::invalid_argument("mymyr SharedEvaluator: max_wait must not be negative");
}

void SharedEvaluator::enter()
{
    std::lock_guard lock(m_mutex);
    ++m_active;
}

void SharedEvaluator::leave()
{
    {
        std::lock_guard lock(m_mutex);
        if (--m_active == 0)
            m_errors.clear();
    }
    m_cv.notify_all();  // the remaining searches may all be waiting now
}

bool SharedEvaluator::due(Clock::time_point now) const
{
    return !m_queue.empty() &&
           (m_queued_rows >= m_max_batch || m_waiting >= m_active || now >= m_queue.front()->since + m_max_wait);
}

std::exception_ptr SharedEvaluator::evaluate(const Request& request)
{
    if (request.states.empty())
        return nullptr;
    std::unique_lock lock(m_mutex);
    Waiting w{&request, Clock::now()};
    m_queue.push_back(&w);
    m_queued_rows += request.states.size();
    ++m_waiting;
    m_cv.notify_all();
    while (!w.done)
    {
        if (!m_calling && due(Clock::now()))
            lead(lock);
        else if (m_calling || m_queue.empty())
            m_cv.wait(lock);
        else
            m_cv.wait_until(lock, m_queue.front()->since + m_max_wait);
    }
    return w.error;
}

void SharedEvaluator::lead(std::unique_lock<std::mutex>& lock)
{
    m_calling = true;
    std::vector<Waiting*> taken;
    std::vector<Request> requests;
    u64 rows = 0;
    while (!m_queue.empty())
    {
        Waiting* q = m_queue.front();
        const u64 k = q->request->states.size();
        if (!taken.empty() && rows + k > m_max_batch)
            break;
        taken.push_back(q);
        requests.push_back(*q->request);
        rows += k;
        m_queue.pop_front();
    }
    m_queued_rows -= rows;
    lock.unlock();
    std::exception_ptr error;
    try
    {
        m_call(requests);
    }
    catch (...)
    {
        error = std::current_exception();
    }
    m_calls.fetch_add(1, std::memory_order_relaxed);
    m_rows.fetch_add(rows, std::memory_order_relaxed);
    u64 largest = m_largest.load(std::memory_order_relaxed);
    while (rows > largest && !m_largest.compare_exchange_weak(largest, rows, std::memory_order_relaxed))
    {
    }
    lock.lock();
    if (error)
        m_errors.push_back(error);
    for (Waiting* q : taken)
    {
        q->done = true;
        q->error = error;
    }
    m_waiting -= static_cast<u32>(taken.size());
    m_calling = false;
    m_cv.notify_all();
}

SharedHeuristic::SharedHeuristic(SharedEvaluator& shared, const void* tag) : m_shared(shared), m_tag(tag) { m_shared.enter(); }

SharedHeuristic::~SharedHeuristic() { m_shared.leave(); }

Value SharedHeuristic::evaluate(StateView s)
{
    Value v = 0;
    evaluate_batch(std::span<const StateView>(&s, 1), std::span<Value>(&v, 1));
    return v;
}

void SharedHeuristic::evaluate_batch(std::span<const StateView> states, std::span<Value> out)
{
    if (const std::exception_ptr error = m_shared.evaluate({states, out, m_tag}))
        std::rethrow_exception(error);
    for (const Value v : out)
    {
        ++m_stats.evaluations;
        if (std::isinf(v) && v > 0)
            ++m_stats.dead_ends;
    }
}
}  // namespace mymyr::heuristics
