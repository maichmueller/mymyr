#include "mymyr/task/slot_map.hpp"

#include "mymyr/task/task.hpp"

#include <algorithm>
#include <bit>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include "mymyr/task/numeric.hpp"

namespace mymyr
{
namespace
{
std::string signature(const formalism::TaskData& data, const formalism::Predicate& p)
{
    return std::string(data.str(p.name)) + "/" + std::to_string(p.arity);
}
}

SlotMap::SlotMap(const Task& source, const Task& target) : m_source(source), m_target(target)
{
    const auto& src = source.data();
    const auto& dst = target.data();
    std::unordered_map<std::string, u32> predicates, objects, numerics;
    for (u32 p = 0; p < dst.predicates.size(); ++p)
        predicates.emplace(signature(dst, dst.predicates[p]), p);
    for (u32 o = 0; o < dst.objects.size(); ++o)
        objects.emplace(dst.str(dst.objects[o].name), o);
    for (const auto& p : src.predicates)
    {
        const auto it = predicates.find(signature(src, p));
        m_predicates.push_back(it == predicates.end() ? missing : it->second);
        if (it == predicates.end())
            m_unmatched.push_back(signature(src, p));
    }
    for (const auto& o : src.objects)
    {
        const auto it = objects.find(std::string(src.str(o.name)));
        m_objects.push_back(it == objects.end() ? missing : it->second);
        if (it == objects.end())
            m_unmatched.emplace_back(src.str(o.name));
    }
    for (u32 n = 0; n < target.numeric_slots(); ++n)
        numerics.emplace(target.numeric_name(n), n);
    for (u32 n = 0; n < source.numeric_slots(); ++n)
    {
        const auto it = numerics.find(source.numeric_name(n));
        m_numeric.push_back(it == numerics.end() ? missing : it->second);
        if (it == numerics.end())
            m_unmatched.push_back(source.numeric_name(n));
    }
    for (const auto& atom : src.static_init)
    {
        std::vector<u32> args;
        for (const auto o : src.objects_of(atom))
            args.push_back(o.v);
        const u32 slot = match_atom(atom.pred.v, args);
        if (slot < checked_static)
        {
            if (slot / 64 >= m_constant.size())
                m_constant.resize(slot / 64 + 1, 0);
            m_constant[slot / 64] |= u64{1} << (slot % 64);
        }
    }
}

std::shared_ptr<SlotMap> SlotMap::between(const Task& source, const Task& target)
{
    return std::make_shared<SlotMap>(source, target);
}

u32 SlotMap::match_atom(u32 predicate, std::span<const u32> arguments) const
{
    const u32 p = m_predicates.at(predicate);
    if (p == missing)
        return missing;
    std::vector<u32> args;
    for (const u32 o : arguments)
    {
        if (m_objects.at(o) == missing)
            return missing;
        args.push_back(m_objects[o]);
    }
    const auto& d = m_target.data();
    const auto kind = d.predicates[p].kind;
    if (kind == formalism::PredKind::Static)
    {
        for (const auto& a : d.static_init)
        {
            if (a.pred.v != p)
                continue;
            const auto objects = d.objects_of(a);
            if (std::equal(args.begin(), args.end(), objects.begin(), objects.end(),
                           [](u32 x, ObjectId y) { return x == y.v; }))
                return checked_static;
        }
        return missing;
    }
    if (kind != formalism::PredKind::Fluent)
        return missing;
    const auto& layout = m_target.atoms().layout();
    const auto cid = layout.encode(p, args.data());
    return cid < layout.total ? m_target.atoms().slot_if_any(cid) : missing;
}

std::shared_ptr<const std::vector<u32>> SlotMap::snapshot() const
{
    std::lock_guard lock(m_mutex);
    const u32 size = m_source.atoms().fluent_slots();
    if (!m_slots || m_slots->size() < size)
    {
        auto slots = std::make_shared<std::vector<u32>>(m_slots ? *m_slots : std::vector<u32>{});
        slots->reserve(size);
        for (u32 slot = static_cast<u32>(slots->size()); slot < size; ++slot)
        {
            const SlotId id{slot};
            slots->push_back(match_atom(m_source.atoms().predicate(id).v, m_source.atoms().arguments(id)));
        }
        m_slots = std::move(slots);
    }
    return m_slots;
}

bool SlotMap::translate_row(std::span<const u32> slots, StateView source, std::span<u64> atoms,
                            std::span<u64> numeric, bool drop, u64 row) const
{
    if (atoms.size() < m_constant.size() || numeric.size() < m_target.numeric_words() ||
        source.nnum < m_source.numeric_words())
        throw std::invalid_argument("mymyr: translate: row is too narrow");
    std::fill(atoms.begin(), atoms.end(), u64{0});
    std::copy(m_constant.begin(), m_constant.end(), atoms.begin());
    std::fill(numeric.begin(), numeric.end(), u64{0});
    for (u32 n = 0; n < m_target.numeric_slots(); ++n)
        plan::store(m_target.numeric(), numeric.data(), n, m_target.numeric().initial[n]);
    bool lossless = true;
    const auto absent = [&](const std::string& name) {
        if (!drop)
            throw std::invalid_argument("mymyr: translate: row " + std::to_string(row) + ": " + name +
                                        " has no representable value in the target");
        lossless = false;
    };
    for (u32 word = 0; word < source.nw; ++word)
        for (u64 bits = source.w[word]; bits; bits &= bits - 1)
        {
            const u64 slot = u64{word} * 64 + std::countr_zero(bits);
            if (slot >= slots.size())
                throw std::invalid_argument("mymyr: translate: row " + std::to_string(row) +
                                            " sets an unknown source slot " + std::to_string(slot));
            const u32 mapped = slots[slot];
            if (mapped == missing)
                absent(m_source.format(SlotId{static_cast<u32>(slot)}));
            else if (mapped != checked_static)
            {
                if (mapped / 64 >= atoms.size())
                    throw std::invalid_argument("mymyr: translate: destination row is too narrow");
                atoms[mapped / 64] |= u64{1} << (mapped % 64);
            }
        }
    for (u32 n = 0; n < m_numeric.size(); ++n)
    {
        const u32 mapped = m_numeric[n];
        if (mapped == missing)
        {
            absent(m_source.numeric_name(n));
            continue;
        }
        const f64 v = plan::load(m_source.numeric(), source.num, n);
        const f64 converted = plan::canonical(m_target.numeric(), v);
        const bool integral = m_target.numeric_storage() != NumericStorage::I32 ||
                              (converted >= -2147483648.0 && converted <= 2147483647.0 && converted == std::nearbyint(converted));
        if (!integral || !(v == converted || (std::isnan(v) && std::isnan(converted))))
        {
            absent(m_source.numeric_name(n));
            continue;
        }
        plan::store(m_target.numeric(), numeric.data(), mapped, v);
    }
    return lossless;
}

State SlotMap::translate(StateView source, bool drop, bool* lossless, u64 row) const
{
    const auto slots = snapshot();
    std::vector<u64> atoms(m_target.words()), numeric(m_target.numeric_words());
    const bool ok = translate_row(*slots, source, atoms, numeric, drop, row);
    if (lossless)
        *lossless = ok;
    return State(atoms.data(), static_cast<u32>(atoms.size()), numeric.data(), static_cast<u32>(numeric.size()));
}

void map_fluent_words(SlotMapView map, std::span<const u64> source, std::span<u64> destination, u32 target_slots)
{
    std::fill(destination.begin(), destination.end(), u64{0});
    for (usize word = 0; word < source.size(); ++word)
        for (u64 bits = source[word]; bits; bits &= bits - 1)
        {
            const u64 slot = word * 64 + std::countr_zero(bits);
            if (slot >= map.size || !map.data)
                throw std::invalid_argument("mymyr: encoding: source atom slot " + std::to_string(slot) +
                                            " is outside the slot map");
            const u32 mapped = map.data[slot];
            if (mapped >= target_slots || mapped / 64 >= destination.size())
                throw std::invalid_argument("mymyr: encoding: source atom slot " + std::to_string(slot) +
                                            " has no slot in the destination");
            destination[mapped / 64] |= u64{1} << (mapped % 64);
        }
}
}  // namespace mymyr
