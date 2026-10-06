#pragma once
// Minimal stand-ins for mimir's hash-consing repositories (mimir `Repositories`, loki `IndexedHashSet`).
//
// Mimir orders nearly every list by the *creation index* of its elements in these repositories. We do not need
// mimir's objects, only those indices, so each repository becomes a `Table`: a key -> dense id map where ids are
// assigned in creation order. A table may have a parent (mimir's problem repositories are children of the
// domain's): lookups check the parent first and child ids continue the parent's id space, exactly like
// `loki::IndexedHashSet`. A parent is never modified while it has children, so concurrent children (one per
// `instantiate` call) only read it.

#include "mymyr/core/types.hpp"

#include <absl/container/flat_hash_set.h>
#include <absl/container/inlined_vector.h>
#include <absl/hash/hash.h>

#include <cassert>
#include <cstring>
#include <deque>
#include <optional>
#include <string>
#include <vector>

namespace mymyr::frontend::detail
{
using Key = absl::InlinedVector<u64, 7>;

template<class K>
class Table
{
public:
    explicit Table(const Table* parent = nullptr) :
        m_parent(parent), m_base(parent ? parent->size() : 0), m_set(0, Hash{this}, Eq{this})
    {
    }
    Table(const Table&) = delete;
    Table& operator=(const Table&) = delete;

    [[nodiscard]] u32 size() const { return m_base + static_cast<u32>(m_items.size()); }
    [[nodiscard]] u32 base() const { return m_base; }  // ids below belong to the parent chain

    [[nodiscard]] std::optional<u32> find(const K& k) const
    {
        for (const Table* t = this; t; t = t->m_parent)
            if (auto it = t->m_set.find(k); it != t->m_set.end())
                return *it;
        return std::nullopt;
    }

    /// Returns the id and whether it was created now.
    std::pair<u32, bool> get_or_create(const K& k)
    {
        if (m_parent)
            if (auto id = m_parent->find(k))
                return {*id, false};
        if (auto it = m_set.find(k); it != m_set.end())
            return {*it, false};
        const u32 id = size();
        m_items.push_back(k);
        m_set.insert(id);
        return {id, true};
    }
    u32 id(const K& k) { return get_or_create(k).first; }

    /// Keys have stable addresses (std::deque), so references stay valid while the table grows.
    [[nodiscard]] const K& operator[](u32 id) const
    {
        assert(id < size());
        return id < m_base ? (*m_parent)[id] : m_items[id - m_base];
    }

private:
    // a flat set of local ids that hashes and compares the keys they stand for (heterogeneous lookup by key)
    struct Hash
    {
        using is_transparent = void;
        const Table* t;
        size_t operator()(u32 id) const { return absl::Hash<K>{}(t->m_items[id - t->m_base]); }
        size_t operator()(const K& k) const { return absl::Hash<K>{}(k); }
    };
    struct Eq
    {
        using is_transparent = void;
        const Table* t;
        bool operator()(u32 a, u32 b) const { return a == b; }
        bool operator()(u32 a, const K& k) const { return t->m_items[a - t->m_base] == k; }
        bool operator()(const K& k, u32 a) const { return t->m_items[a - t->m_base] == k; }
    };

    const Table* m_parent;
    u32 m_base;
    std::deque<K> m_items;
    absl::flat_hash_set<u32, Hash, Eq> m_set;
};

inline u64 f64_bits(f64 v)
{
    u64 b;
    std::memcpy(&b, &v, sizeof b);
    return b;
}
inline f64 bits_f64(u64 b)
{
    f64 v;
    std::memcpy(&v, &b, sizeof v);
    return v;
}

// Terms inside keys: objects are tagged with the top bit, variables are variable-table ids.
constexpr u64 kObjectTag = u64{1} << 63;
[[nodiscard]] constexpr u64 obj_term(u32 o) { return kObjectTag | o; }
[[nodiscard]] constexpr bool is_obj_term(u64 t) { return (t & kObjectTag) != 0; }
[[nodiscard]] constexpr u32 term_obj(u64 t) { return static_cast<u32>(t & ~kObjectTag); }

// Function expression tags (first key word).
enum : u64
{
    kFeNumber = 1,
    kFeBinary,
    kFeMulti,
    kFeMinus,
    kFeFunction,
};

/// One level of mimir's repositories. Separate index spaces exist where mimir compares indices (per entity
/// type); index spaces mimir never compares across (e.g. static vs fluent literals) may share a table because
/// sharing preserves the relative creation order inside each kind.
struct Repo
{
    /// What mimir derives when it creates a conjunctive condition: its nullary literals as ground literals
    /// (per kind, sorted by ground-literal index) and its nullary numeric constraints grounded (sorted).
    struct CondExtra
    {
        std::vector<u32> nullary[3];
        std::vector<u32> gncs;
    };

    explicit Repo(const Repo* parent = nullptr) :
        parent_repo(parent),
        names(parent ? &parent->names : nullptr),
        var(parent ? &parent->var : nullptr),
        param(parent ? &parent->param : nullptr),
        lit(parent ? &parent->lit : nullptr),
        glit(parent ? &parent->glit : nullptr),
        fe(parent ? &parent->fe : nullptr),
        gnum(parent ? &parent->gnum : nullptr),
        gbin(parent ? &parent->gbin : nullptr),
        gmulti(parent ? &parent->gmulti : nullptr),
        gminus(parent ? &parent->gminus : nullptr),
        gfunc{Table<Key>(parent ? &parent->gfunc[0] : nullptr), Table<Key>(parent ? &parent->gfunc[1] : nullptr),
              Table<Key>(parent ? &parent->gfunc[2] : nullptr)},
        gfe(parent ? &parent->gfe : nullptr),
        nc(parent ? &parent->nc : nullptr),
        gnc(parent ? &parent->gnc : nullptr),
        ne(parent ? &parent->ne : nullptr),
        nea(parent ? &parent->nea : nullptr),
        cond(parent ? &parent->cond : nullptr),
        ceff(parent ? &parent->ceff : nullptr),
        ce(parent ? &parent->ce : nullptr),
        axiom(parent ? &parent->axiom : nullptr),
        gfv{Table<Key>(parent ? &parent->gfv[0] : nullptr), Table<Key>(parent ? &parent->gfv[1] : nullptr),
            Table<Key>(parent ? &parent->gfv[2] : nullptr)}
    {
    }
    Repo(const Repo&) = delete;
    Repo& operator=(const Repo&) = delete;

    [[nodiscard]] const CondExtra& extra(u32 cond_id) const
    {
        return cond_id < cond.base() ? parent_repo->extra(cond_id) : cond_extra[cond_id - cond.base()];
    }

    const Repo* parent_repo;
    Table<std::string> names;  // interned variable names
    Table<Key> var;            // [name, parameter_index]
    Table<Key> param;          // [var, type ids...]
    Table<Key> lit;            // [pred, polarity, terms...]
    Table<Key> glit;           // [pred, polarity, objects...]
    Table<Key> fe;             // lifted function expressions: [tag, payload...] (index = mimir's wrapper index)
    // ground function expressions: inner tables keep their own index spaces (mimir sorts multi-operator children
    // by the inner index), `gfe` is the wrapper [tag, inner id]
    Table<Key> gnum, gbin, gmulti, gminus;
    Table<Key> gfunc[3];  // per function kind (static, fluent, auxiliary): [function, objects...]
    Table<Key> gfe;
    Table<Key> nc;       // [comparator, lhs fe, rhs fe, arity]
    Table<Key> gnc;      // [comparator, lhs gfe, rhs gfe]
    Table<Key> ne;       // fluent numeric effect [op, function, nterms, terms..., fe]
    Table<Key> nea;      // auxiliary numeric effect [op, function, nterms, terms..., fe]
    Table<Key> cond;     // conjunctive condition [np, params..., n, static lits..., n, fluent..., n, derived..., n, ncs...]
    Table<Key> ceff;     // conjunctive effect [np, params..., n, lits..., n, nes..., has_aux, aux]
    Table<Key> ce;       // conditional effect [cond, ceff]
    Table<Key> axiom;    // [cond, head literal]
    Table<Key> gfv[3];   // ground function values per kind: [function, objects..., value bits]
    std::vector<CondExtra> cond_extra;  // for the conditions created at this level
};
}  // namespace mymyr::frontend::detail
