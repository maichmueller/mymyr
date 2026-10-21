#pragma once
// The handle registry of the XLA FFI targets (rl_jax_bindings.cpp: environments, encoding_jax_bindings.cpp: encoders).
// FFI attributes are scalars, so the native object a call works on enters it as an int64 handle into one process-wide
// table. Handles are never reused; a call whose handle is gone fails with an error.

#include "mymyr/core/types.hpp"

#include <memory>
#include <mutex>
#include <unordered_map>

namespace mymyr::python
{
/// A registered object. The registry drops its reference outside its lock, so an entry may synchronize a device when
/// it is destroyed.
struct FfiEntry
{
    virtual ~FfiEntry() = default;
};

class FfiRegistry
{
public:
    i64 add(std::shared_ptr<FfiEntry> e)
    {
        std::lock_guard lock(m_mutex);
        const i64 h = m_next++;
        m_entries.emplace(h, std::move(e));
        return h;
    }
    void remove(i64 h)
    {
        std::shared_ptr<FfiEntry> keep;  // destroyed after the lock
        {
            std::lock_guard lock(m_mutex);
            auto it = m_entries.find(h);
            if (it == m_entries.end())
                return;
            keep = std::move(it->second);
            m_entries.erase(it);
        }
    }
    [[nodiscard]] std::shared_ptr<FfiEntry> find(i64 h)
    {
        std::lock_guard lock(m_mutex);
        auto it = m_entries.find(h);
        return it == m_entries.end() ? nullptr : it->second;
    }
    /// The entry under `h` if it is a T, else null.
    template<class T>
    [[nodiscard]] std::shared_ptr<T> find_as(i64 h)
    {
        return std::dynamic_pointer_cast<T>(find(h));
    }

private:
    std::mutex m_mutex;
    std::unordered_map<i64, std::shared_ptr<FfiEntry>> m_entries;
    i64 m_next = 1;
};

/// The process-wide registry. Never destroyed: XLA may call a handler during interpreter shutdown.
inline FfiRegistry& ffi_registry()
{
    static FfiRegistry* r = new FfiRegistry();
    return *r;
}
}  // namespace mymyr::python
