#include "mymyr/task/slot_map.hpp"

#include "mymyr/core/hash.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/numeric.hpp"

#include <algorithm>
#include <bit>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace mymyr
{
namespace
{
std::string signature(const formalism::TaskData& data, const formalism::Predicate& p)
{
    return std::string(data.str(p.name)) + "/" + std::to_string(p.arity);
}

bool same_layout(const CanonicalLayout& a, const CanonicalLayout& b)
{
    return a.num_objects == b.num_objects && a.fluent_count == b.fluent_count && a.total == b.total &&
           a.arity == b.arity && a.offset == b.offset && a.size == b.size && a.pos_begin == b.pos_begin &&
           a.stride == b.stride && a.rs == b.rs;
}

std::vector<u32> fact(const formalism::TaskData& data, const formalism::GroundAtom& atom, u32 predicate)
{
    std::vector<u32> key{predicate};
    for (const auto o : data.objects_of(atom))
        key.push_back(o.v);
    return key;
}

/// Throws for a bit at or past slot `slots`; returns the words that hold known slots.
u32 check_slots(usize slots, StateView source, u64 row)
{
    const u32 full = static_cast<u32>(slots / 64);
    for (u32 word = full; word < source.nw; ++word)
    {
        const u64 known = word == full ? (u64{1} << (slots % 64)) - 1 : 0;
        if (const u64 unknown = source.w[word] & ~known)
            throw std::invalid_argument("mymyr: translate: row " + std::to_string(row) + " sets an unknown source slot " +
                                        std::to_string(u64{word} * 64 + std::countr_zero(unknown)));
    }
    return std::min(source.nw, full + 1);
}
}

usize SlotMap::FactHash::operator()(const std::vector<u32>& fact) const noexcept
{
    u64 h = fact.size();
    for (const u32 v : fact)
        h = hash::combine(h, v);
    return static_cast<usize>(h);
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
    bool identity = src.predicates.size() == dst.predicates.size() && src.objects.size() == dst.objects.size();
    for (u32 p = 0; p < src.predicates.size(); ++p)
    {
        const auto it = predicates.find(signature(src, src.predicates[p]));
        m_predicates.push_back(it == predicates.end() ? missing : it->second);
        identity = identity && m_predicates.back() == p;
        if (it == predicates.end())
            m_unmatched.push_back(signature(src, src.predicates[p]));
    }
    for (u32 o = 0; o < src.objects.size(); ++o)
    {
        const auto it = objects.find(std::string(src.str(src.objects[o].name)));
        m_objects.push_back(it == objects.end() ? missing : it->second);
        identity = identity && m_objects.back() == o;
        if (it == objects.end())
            m_unmatched.emplace_back(src.str(src.objects[o].name));
    }
    // Equal predicates, objects and layouts give every atom the same canonical id in both tasks, which holds for
    // twins of one parse and for repeated parses of one text.
    m_same_layout = identity && same_layout(source.atoms().layout(), target.atoms().layout());
    for (u32 n = 0; n < target.numeric_slots(); ++n)
        numerics.emplace(target.numeric_name(n), n);
    for (u32 n = 0; n < source.numeric_slots(); ++n)
    {
        const auto it = numerics.find(source.numeric_name(n));
        m_numeric.push_back(it == numerics.end() ? missing : it->second);
        if (it == numerics.end())
            m_unmatched.push_back(source.numeric_name(n));
    }
    // Source fluents of a predicate that is static in the target are true exactly when the target holds the fact.
    std::vector<bool> checked(dst.predicates.size(), false);
    bool any_checked = false;
    for (u32 p = 0; p < src.predicates.size(); ++p)
        if (src.predicates[p].kind == formalism::PredKind::Fluent && m_predicates[p] != missing &&
            dst.predicates[m_predicates[p]].kind == formalism::PredKind::Static)
        {
            checked[m_predicates[p]] = true;
            any_checked = true;
        }
    if (any_checked)
        for (const auto& atom : dst.static_init)
            if (checked[atom.pred.v])
                m_target_statics.insert(fact(dst, atom, atom.pred.v));
    // Only source static facts of a predicate that is fluent in the target affect rows: they become constant bits, or
    // are reported missing. A static or derived target predicate, or none, needs neither.
    for (const auto& atom : src.static_init)
    {
        const u32 p = m_predicates[atom.pred.v];
        if (p == missing || dst.predicates[p].kind != formalism::PredKind::Fluent)
            continue;
        std::vector<u32> args;
        for (const auto o : src.objects_of(atom))
            args.push_back(o.v);
        const u32 entry = match_atom(atom.pred.v, args);
        if (entry != missing)
        {
            // Constant bits hold in every translated row, so a lazy target interns them here.
            const u32 slot = target.atoms().frozen() ? entry : target.atoms().intern(entry);
            if (slot / 64 >= m_constant.size())
                m_constant.resize(slot / 64 + 1, 0);
            m_constant[slot / 64] |= u64{1} << (slot % 64);
        }
        else
        {
            std::string name = "(" + std::string(src.str(src.predicates[atom.pred.v].name));
            for (const auto o : src.objects_of(atom))
                name += " " + source.object_name(o);
            m_missing_constants.push_back(name + ")");
        }
    }
}

std::shared_ptr<SlotMap> SlotMap::between(const Task& source, const Task& target)
{
    return std::make_shared<SlotMap>(source, target);
}

u32 SlotMap::target_entry(u32 cid) const
{
    const auto& atoms = m_target.atoms();
    if (cid >= atoms.layout().total)
        return missing;
    return atoms.frozen() ? atoms.find(cid) : cid;
}

u32 SlotMap::match_atom(u32 predicate, std::span<const u32> arguments) const
{
    const u32 p = m_predicates.at(predicate);
    if (p == missing)
        return missing;
    std::vector<u32> args;
    args.reserve(arguments.size() + 1);
    args.push_back(p);
    for (const u32 o : arguments)
    {
        if (m_objects.at(o) == missing)
            return missing;
        args.push_back(m_objects[o]);
    }
    const auto kind = m_target.data().predicates[p].kind;
    if (kind == formalism::PredKind::Static)
        return m_target_statics.contains(args) ? checked_static : missing;
    if (kind != formalism::PredKind::Fluent)
        return missing;
    const auto cid = m_target.atoms().layout().encode(p, args.data() + 1);
    return cid < m_target.atoms().layout().total ? target_entry(static_cast<u32>(cid)) : missing;
}

std::shared_ptr<const std::vector<u32>> SlotMap::snapshot() const
{
    std::lock_guard lock(m_mutex);
    const auto& atoms = m_source.atoms();
    const u32 size = atoms.fluent_slots();
    if (!m_slots || m_slots->size() < size)
    {
        auto slots = std::make_shared<std::vector<u32>>(m_slots ? *m_slots : std::vector<u32>{});
        slots->reserve(size);
        for (u32 slot = static_cast<u32>(slots->size()); slot < size; ++slot)
        {
            // A reserved lazy slot's record becomes readable when its canonical id is published.
            const CanonicalAtom cid = atoms.canonical(AtomKind::Fluent, slot);
            if (m_same_layout)
                slots->push_back(target_entry(static_cast<u32>(cid)));
            else
            {
                const SlotId id{slot};
                slots->push_back(match_atom(atoms.predicate(id).v, atoms.arguments(id)));
            }
        }
        m_slots = std::move(slots);
    }
    return m_slots;
}

void SlotMap::prepare_row(std::span<const u32> slots, StateView source, u64 row) const
{
    const u32 known = check_slots(slots.size(), source, row);
    if (m_target.atoms().frozen())
        return;
    for (u32 word = 0; word < known; ++word)
        for (u64 bits = source.w[word]; bits; bits &= bits - 1)
            if (const u32 entry = slots[u64{word} * 64 + std::countr_zero(bits)]; entry < checked_static)
                (void)m_target.atoms().intern(entry);
}

bool SlotMap::translate_row(std::span<const u32> slots, StateView source, std::span<u64> atoms,
                            std::span<u64> numeric, bool drop, u64 row) const
{
    if (atoms.size() < m_constant.size() || numeric.size() < m_target.numeric_words() ||
        source.nnum < m_source.numeric_words())
        throw std::invalid_argument("mymyr: translate: row is too narrow");
    const u32 known = check_slots(slots.size(), source, row);
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
    const bool lazy = !m_target.atoms().frozen();
    for (u32 word = 0; word < known; ++word)
        for (u64 bits = source.w[word]; bits; bits &= bits - 1)
        {
            const u64 slot = u64{word} * 64 + std::countr_zero(bits);
            const u32 entry = slots[slot];
            if (entry == missing)
                absent(m_source.format(SlotId{static_cast<u32>(slot)}));
            else if (entry != checked_static)
            {
                const u32 mapped = lazy ? m_target.atoms().intern(entry) : entry;
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
    for (const auto& name : m_missing_constants)
        absent(name);
    return lossless;
}

State SlotMap::translate(StateView source, bool drop, bool* lossless, u64 row) const
{
    const auto slots = snapshot();
    prepare_row(*slots, source, row);
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
