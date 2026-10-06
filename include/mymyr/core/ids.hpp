#pragma once
// Strongly typed u32 ids. Every core entity is an index into a flat array, and the tag keeps object ids from being
// passed where slot ids are expected. Ids are trivially copyable and have the same layout as a u32, so arrays of ids
// can be handed to kernels and to NumPy as uint32.

#include "mymyr/core/types.hpp"

#include <compare>
#include <functional>
#include <type_traits>

namespace mymyr
{
template<class Tag>
struct Id
{
    static constexpr u32 invalid_value = ~u32{0};

    u32 v = invalid_value;

    constexpr Id() = default;
    constexpr explicit Id(u32 value) : v(value) {}

    [[nodiscard]] constexpr bool valid() const { return v != invalid_value; }
    [[nodiscard]] constexpr u32 value() const { return v; }
    [[nodiscard]] static constexpr Id invalid() { return Id{}; }

    friend constexpr auto operator<=>(Id, Id) = default;
};

struct ObjectTag;
struct TypeTag;
struct PredicateTag;
struct SlotTag;       // bit position of an atom inside a state
struct SchemaTag;
struct AxiomTag;
struct FunctionTag;
struct StateTag;      // id inside a user-created container; states themselves are values

using ObjectId = Id<ObjectTag>;
using TypeId = Id<TypeTag>;
using PredicateId = Id<PredicateTag>;
using SlotId = Id<SlotTag>;
using SchemaId = Id<SchemaTag>;
using AxiomId = Id<AxiomTag>;
using FunctionId = Id<FunctionTag>;
using StateId = Id<StateTag>;

/// Canonical atom id: typed-dense mixed radix over per-position ranks. Portable across parses and processes.
using CanonicalAtom = u64;

static_assert(std::is_trivially_copyable_v<ObjectId> && sizeof(ObjectId) == sizeof(u32));
}  // namespace mymyr

template<class Tag>
struct std::hash<mymyr::Id<Tag>>
{
    std::size_t operator()(mymyr::Id<Tag> id) const noexcept { return std::hash<mymyr::u32>{}(id.v); }
};
