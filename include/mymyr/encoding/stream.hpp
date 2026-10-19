#pragma once

// Streams of rows for the relational encoder: rows and their per-row inputs are appended over time, and a flush encodes
// them in one call. A stream copies what it is given, in the layout EncodeInput takes (fixed-width rows of state words,
// goal masks and derived words, and the action rows of an expansion), so a flush reads its storage directly (or
// gathers it row by row, for a mutable stream) and the caller may reuse its buffers at once. The bundle of a flush is
// the bundle encode() returns for the same rows in the same order, bit for bit, whatever the order of the appends.
//
// Per-row inputs. The rows of a stream carry the same inputs: a stream without rows takes any set of them (derived
// words, goals as a pair of masks or with the derived pair, actions) with its first append, and from then on each
// append must give exactly that set. A stream that holds no rows again (after a clear or a flush of the append-only
// stream, or once every row of a mutable stream is removed) takes any set again. Derived words and derived goal masks
// are only stored when some instance of the table has derived words.
//
// Action rows. An appended expansion row names the state it belongs to; the stream stores the labels of the rows
// (schema and binding) of each state and nothing else. A flush encodes with one expansion made of the stored rows,
// graph after graph, so the entity_ref of an action entity reads as encode() gives it for an expansion of the same states.
//
// Errors. Invalid rows raise std::invalid_argument ("mymyr: encoding: ...") when they are appended where the rows alone
// show it (a task id out of range, a row narrower than its instance, goal masks narrower than their instance, a
// malformed expansion); the rest (a set bit the instance has no atom for, labels that are not an instance's, actions out
// of canonical order) is the encoder's and raises at the flush. A call that raises changes nothing: an append or update
// stores none of its rows, and a flush that raises keeps the rows.
//
// Threads. Every call locks the stream: calls from several threads are serialized, and a flush holds the stream while
// it encodes.

#include "mymyr/core/thread_pool.hpp"
#include "mymyr/encoding/encoder.hpp"

#include <memory>
#include <span>
#include <vector>

namespace mymyr::encoding
{
/// An append-only stream: rows are appended, a flush encodes everything appended since the last flush and empties the
/// stream.
class FlatStream
{
public:
    explicit FlatStream(std::shared_ptr<const FlatEncoder> encoder);
    ~FlatStream();
    FlatStream(FlatStream&&) noexcept;
    FlatStream& operator=(FlatStream&&) noexcept;

    [[nodiscard]] const std::shared_ptr<const FlatEncoder>& encoder() const noexcept;

    /// Appends the `rows.states.rows` rows of `rows` with their per-row inputs (task_ids, derived, goals, actions;
    /// the pointers of `rows` are read during the call only). Throws std::invalid_argument for rows that are invalid
    /// or give other inputs than the rows already held; the stream is then unchanged.
    void append(const EncodeInput& rows);
    /// The rows held.
    [[nodiscard]] u64 size() const;
    /// Drops every row.
    void clear();

    /// Encodes the rows held and empties the stream, on the members of `pool` (as FlatEncoder::encode). If the
    /// encoding throws, the rows are kept.
    [[nodiscard]] std::shared_ptr<const rl::ArrayBundle> flush(ThreadPool& pool);
    /// The capacity form (FlatEncoder::encode_capacity): the rows are dropped iff the batch fits (sizes.overflow == 0),
    /// so that a flush that overflowed can be repeated with larger capacities.
    [[nodiscard]] std::shared_ptr<const rl::ArrayBundle> flush(ThreadPool& pool, const BatchShape& capacities,
                                                               FlatSizes& sizes);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

/// A stream of rows under indices the caller chooses (any i64), in which rows can be replaced and removed. A flush
/// encodes the rows held in ascending order of their indices and keeps them: the graph j of the batch is the row of
/// indices()[j].
class FlatMutableStream
{
public:
    explicit FlatMutableStream(std::shared_ptr<const FlatEncoder> encoder);
    ~FlatMutableStream();
    FlatMutableStream(FlatMutableStream&&) noexcept;
    FlatMutableStream& operator=(FlatMutableStream&&) noexcept;

    [[nodiscard]] const std::shared_ptr<const FlatEncoder>& encoder() const noexcept;

    /// Stores row k of `rows` (see FlatStream::append) under indices[k]. Throws std::invalid_argument if an index is held
    /// already or repeats, or the rows are invalid; the stream is then unchanged.
    void append(std::span<const i64> indices, const EncodeInput& rows);
    /// Replaces the rows under `indices` with the rows of `rows` (all inputs of the stream are given anew). Throws
    /// std::out_of_range if an index is not held, std::invalid_argument if one repeats or the rows are invalid; the
    /// stream is then unchanged.
    void update(std::span<const i64> indices, const EncodeInput& rows);
    /// Removes the rows under `indices`. Throws std::out_of_range if an index is not held and std::invalid_argument if
    /// one repeats; the stream is then unchanged.
    void remove(std::span<const i64> indices);
    [[nodiscard]] bool contains(i64 index) const;
    /// The rows held.
    [[nodiscard]] u64 size() const;
    /// Drops every row.
    void clear();
    /// The indices of the rows held, ascending: the order of the graphs of a flush.
    [[nodiscard]] std::vector<i64> indices() const;

    /// Encodes the rows held, in ascending order of their indices, on the members of `pool`. The rows are kept.
    [[nodiscard]] std::shared_ptr<const rl::ArrayBundle> flush(ThreadPool& pool);
    /// The capacity form (FlatEncoder::encode_capacity). The rows are kept.
    [[nodiscard]] std::shared_ptr<const rl::ArrayBundle> flush(ThreadPool& pool, const BatchShape& capacities,
                                                               FlatSizes& sizes);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
}  // namespace mymyr::encoding
