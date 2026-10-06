#include "mymyr/task/atom_index.hpp"

#include <algorithm>
#include <atomic>
#include <stdexcept>

namespace mymyr
{
u32 CanonicalLayout::decode(CanonicalAtom c, u32* args) const
{
    // by_offset is sorted by offset; find the last predicate whose offset is <= c
    auto it = std::upper_bound(by_offset.begin(), by_offset.end(), c, [&](CanonicalAtom x, u32 p) { return x < offset[p]; });
    if (it == by_offset.begin() || c >= total)
        throw std::out_of_range("CanonicalLayout::decode: id out of range");
    const u32 p = *(it - 1);
    u64 r = c - offset[p];
    for (u32 i = 0; i < arity[p]; ++i)
    {
        const u64 pos = pos_begin[p] + i;
        const u64 q = r / stride[pos];
        r -= q * stride[pos];
        args[i] = domain[domain_begin[pos] + q];
    }
    return p;
}

ViewOp view_op_of(const ViewLayout& v, u32 pred, u32 arity, const u32* args)
{
    ViewOp op;
    if (arity == 1)
    {
        op.row1 = v.unary_row[pred];
        op.bit1 = args[0];
    }
    else if (arity == 2)
    {
        op.row1 = v.fwd_row[pred] + args[0];
        op.bit1 = args[1];
        op.row2 = v.bwd_row[pred] + args[1];
        op.bit2 = args[0];
    }
    return op;
}

void AtomIndex::init(const CanonicalLayout* layout, const ViewLayout* view, AtomMode mode)
{
    m_layout = layout;
    m_view = view;
    reset(mode);
}

void AtomIndex::reset(AtomMode mode)
{
    const CanonicalLayout& L = *m_layout;
    if (L.total > CanonicalLayout::k_max_total)
        throw std::length_error("AtomIndex: typed-dense canonical space of " + std::to_string(L.total) +
                                " atoms exceeds 2^28; hashed interning is not implemented yet");
    m_mode = mode;
    m_slot_of = std::make_unique<std::atomic<u32>[]>(L.total + 1);
    const u64 counts[2] = {L.fluent_count, L.total - L.fluent_count};
    const u32 stride = 1 + std::max<u32>(1, L.max_arity);
    for (usize k = 0; k < 2; ++k)
    {
        m_records[k].reset(std::max<u64>(1, counts[k]), stride);
        m_ops[k].reset(std::max<u64>(1, counts[k]), 1);
        m_cid[k].reset(std::max<u64>(1, counts[k]), 1);
    }
    m_slot_of[L.total].store(k_empty, std::memory_order_relaxed);
    if (mode == AtomMode::Lazy)
    {
        for (u64 c = 0; c < L.total; ++c)
            m_slot_of[c].store(k_empty, std::memory_order_relaxed);
        m_next[0].store(0, std::memory_order_relaxed);
        m_next[1].store(0, std::memory_order_relaxed);
    }
    else
    {
        for (u64 c = 0; c < L.total; ++c)
        {
            const AtomKind k = L.kind_of(c);
            const u32 s = static_cast<u32>(k == AtomKind::Fluent ? c : c - L.fluent_count);
            write_slot(k, s, c);
            m_slot_of[c].store(s, std::memory_order_relaxed);
        }
        m_next[0].store(static_cast<u32>(counts[0]), std::memory_order_relaxed);
        m_next[1].store(static_cast<u32>(counts[1]), std::memory_order_relaxed);
    }
    // No fence: reset() runs while the task is built, before it is shared; publishing the task to other threads
    // (thread start, mutex, atomic shared_ptr) orders these stores.
}

void AtomIndex::write_slot(AtomKind k, u32 slot, CanonicalAtom c) const
{
    const CanonicalLayout& L = *m_layout;
    u32* rec = m_records[idx(k)].ensure(slot);
    const u32 p = L.decode(c, rec + 1);
    rec[0] = p;
    *m_ops[idx(k)].ensure(slot) = view_op_of(*m_view, p, L.arity[p], rec + 1);
    *m_cid[idx(k)].ensure(slot) = c;
}

u32 AtomIndex::intern_slow(CanonicalAtom c) const
{
    if (c >= m_layout->total)
        throw std::logic_error("AtomIndex::intern: atom outside the typed-dense domains (compiler invariant violated)");
    std::atomic<u32>& e = m_slot_of[c];
    u32 v = e.load(std::memory_order_acquire);
    for (u32 spins = 0;;)
    {
        if (v < k_pending)
            return v;
        if (v == k_empty)
        {
            if (e.compare_exchange_weak(v, k_pending, std::memory_order_acq_rel, std::memory_order_acquire))
            {
                const AtomKind k = m_layout->kind_of(c);
                const u32 s = m_next[idx(k)].fetch_add(1, std::memory_order_acq_rel);
                write_slot(k, s, c);
                e.store(s, std::memory_order_release);
                e.notify_all();  // no syscall unless a thread blocks below
                return s;
            }
            continue;
        }
        // Pending: another thread is writing the slot's records. Spin briefly, then block until it publishes.
        // Yield-spinning here starves a preempted writer once threads outnumber cores (64 Python threads, TSan).
        if (++spins > 64)
            e.wait(k_pending, std::memory_order_acquire);
        v = e.load(std::memory_order_acquire);
    }
}

u64 AtomIndex::bytes() const noexcept
{
    u64 b = (m_layout ? m_layout->total + 1 : 0) * sizeof(u32);
    for (usize k = 0; k < 2; ++k)
        b += m_records[k].bytes() + m_ops[k].bytes() + m_cid[k].bytes();
    return b;
}
}  // namespace mymyr
