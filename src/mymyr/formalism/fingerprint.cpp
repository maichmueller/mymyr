#include "mymyr/formalism/fingerprint.hpp"

#include "mymyr/core/hash.hpp"
#include "mymyr/formalism/text_format.hpp"

#include <cstring>
#include <string_view>

namespace mymyr::formalism
{
namespace
{
u64 hash_bytes(std::string_view s, u64 seed)
{
    u64 h = hash::combine(seed, s.size());
    usize i = 0;
    for (; i + 16 <= s.size(); i += 16)
    {
        u64 w[2];
        std::memcpy(w, s.data() + i, 16);
        h = hash::mulfold(h ^ w[0] ^ hash::k_p1, w[1] ^ hash::k_p2);
    }
    u64 w[2] = {0, 0};
    std::memcpy(w, s.data() + i, s.size() - i);
    return hash::mix64(hash::mulfold(h ^ w[0] ^ hash::k_p1, w[1] ^ hash::k_p2));
}
}  // namespace

u64 fingerprint(const TaskData& t)
{
    u64 h = hash_bytes(write_task_text(t), 0x6d796d7972ULL);
    h = hash::combine(h, hash_bytes(t.domain_name, 1));
    h = hash::combine(h, hash_bytes(t.problem_name, 2));
    for (const Object& o : t.objects)
        h = hash::combine(h, hash_bytes(t.str(o.name), o.constant ? 3 : 4));
    for (const Type& ty : t.types)
        h = hash::combine(h, hash_bytes(t.str(ty.name), 5));
    for (const Schema& s : t.schemas)
        h = hash::combine(h, s.original_arity);
    return h;
}
}  // namespace mymyr::formalism
