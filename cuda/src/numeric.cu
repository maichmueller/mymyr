#include "mymyr/cuda/numeric_kernels.hpp"

#include "lifted_device.cuh"

namespace mymyr::cuda::numeric
{
namespace
{
using namespace rl::dev;
using namespace lifted;
constexpr u32 block = 128;

unsigned grid(u64 n) { return static_cast<unsigned>((n + block - 1) / block); }

/// Numeric effect families persist even across a non-firing conditional effect, matching CPU applicability.
template<u32 OW>
struct ActionWriter
{
    const TaskView& t;
    const Parents& p;
    const Views& v;
    const SuccessorRows& out;
    const u64* row;
    const u64* derived;
    u32 schema, parent;
    u32* error;
    u64 values[k_max_words];
    u8 family[k_max_words];
    u8 touched[k_max_words];
    u64 adds[k_max_words], dels[k_max_words];
    u8 aux_family = 0;
    bool valid = true;

    __device__ bool group(u32 id, const u32* bind)
    {
        const auto& n = t.numeric;
        const u32* g = n.groups + u64{id} * 5;
        for (u32 i = 0; i < g[1]; ++i)
        {
            const u32* e = n.effects + u64{g[0] + i} * 8;
            const u32 slot = e[2] ? static_cast<u32>(numeric_lookup(n, e[3], numeric_key(n, e[3], e[4], e[5], t.num_objects, bind)))
                                 : e[1];
            if (slot == k_none)
                return false;
            const u8 f = e[0] == 0 ? 1 : e[0] <= 2 ? 2 : 3;
            if (family[slot] && (family[slot] != f || f == 1))
                return false;
            family[slot] = f;
            const f64 value = numeric_eval(n, e[6], e[7], row + p.words, bind, t.num_objects);
            if (std::isnan(value))
                return false;
            touched[slot] = 1;
            values[slot] = numeric_bits(numeric_assign(e[0], numeric_value(values, slot), value));
        }
        if (g[2] != k_none)
        {
            const u8 f = g[2] == 0 ? 1 : g[2] <= 2 ? 2 : 3;
            if (aux_family && (aux_family != f || f == 1))
                return false;
            aux_family = f;
            if (std::isnan(numeric_eval(n, g[3], g[4], row + p.words, bind, t.num_objects)))
                return false;
        }
        return true;
    }

    __device__ void literal(u32 begin, u32 count, const u32* bind, bool add)
    {
        for (u32 i = 0; i < count; ++i)
        {
            const u64 key = pattern_key(t, begin + i, bind);
            const u32 slot = slot_of(t, key);
            if (slot == k_none)
            {
                if (add && out.missing_bits && key < t.fluent_total)
                {
                    atomicOr(out.missing_bits + (key >> 5), 1u << (key & 31));
                    atomicOr(out.missing_flag, 1u);
                }
                continue;
            }
            (add ? adds : dels)[slot >> 6] |= u64{1} << (slot & 63);
        }
    }

    __device__ void ce_literals(u32 id, const u32* bind)
    {
        const u32* ce = t.cond_effect + u64{id} * 5;
        literal(ce[1], ce[2], bind, true);
        literal(ce[3], ce[4], bind, false);
    }

    __device__ void conditional(u32 id, u32* bind, bool numeric)
    {
        const u32* ce = t.cond_effect + u64{id} * 5;
        const u32* meta = t.numeric.ce + u64{id} * 3;
        const u32* mx = t.matcher + u64{ce[0]} * k_mc_count;
        const StateRef state{row, p.words, derived, p.derived_words, row + p.words};
        const u64* view = v.data + u64{parent} * v.words;
        if (numeric && !meta[2])
        {
            u64 saved[k_max_words];
            u8 saved_touch[k_max_words];
            for (u32 i = 0; i < t.numeric.slots; ++i)
            {
                saved[i] = values[i];
                saved_touch[i] = touched[i];
            }
            const bool ok = group(meta[0], bind);
            bool fires = false;
            auto emit = [&](const u32* b) { fires = true; ce_literals(id, b); };
            run_fixed<OW, k_ce_depth, false, false, true>(t, mx, state, view, bind, emit);
            if (fires)
                valid &= ok;
            else
                for (u32 i = 0; i < t.numeric.slots; ++i)
                {
                    values[i] = saved[i];
                    touched[i] = saved_touch[i];
                }
        }
        else
        {
            auto emit = [&](const u32* b)
            {
                if (!valid)
                    return;
                if (numeric)
                    valid = group(meta[0], b);
                if (valid)
                    ce_literals(id, b);
            };
            run_fixed<OW, k_ce_depth, false, false, true>(t, mx, state, view, bind, emit);
        }
    }

    __device__ bool collect(const u32* binding)
    {
        u32 bind[k_max_depth];
        const u32* sc = t.schema + u64{schema} * k_sc_count;
        for (u32 i = 0; i < sc[k_sc_arity]; ++i)
            bind[i] = binding[i];
        for (u32 i = 0; i < t.numeric.slots; ++i)
        {
            values[i] = row[p.words + i];
            family[i] = touched[i] = 0;
        }
        for (u32 i = 0; i < k_max_words; ++i)
            adds[i] = dels[i] = 0;
        valid = true;
        aux_family = 0;
        literal(sc[k_sc_adds], sc[k_sc_adds_n], bind, true);
        literal(sc[k_sc_dels], sc[k_sc_dels_n], bind, false);
        const u32* ns = t.numeric.schema + u64{schema} * 2;
        for (u32 i = 0; i < ns[1] && valid; ++i)
        {
            const u32* ref = t.numeric.order + u64{ns[0] + i} * 2;
            if (ref[0])
                conditional(ref[1], bind, true);
            else
                valid = group(ref[1], bind);
        }
        if (!valid)
            return false;
        for (u32 i = 0; i < sc[k_sc_ces_n]; ++i)
        {
            const u32 id = sc[k_sc_ces] + i;
            if (!t.numeric.ce[u64{id} * 3 + 1])
                conditional(id, bind, false);
        }
        for (u32 i = 0; i < t.numeric.slots; ++i)
            if (touched[i])
            {
                const f64 value = numeric_canonical(t.numeric, numeric_value(values, i));
                if (t.numeric.storage && (!(value >= -2147483648.0 && value <= 2147483647.0) || value != std::nearbyint(value)))
                    atomicOr(error, 1u);
                values[i] = numeric_bits(value);
            }
        return true;
    }

    __device__ void put(u64 position)
    {
        const u32 atoms = out.out_words - t.numeric.slots;
        u32 needed = 0;
        for (u32 w = 0; w < k_max_words; ++w)
        {
            const u64 value = ((w < p.words ? row[w] : 0) & ~dels[w]) | adds[w];
            if (value)
                needed = w + 1;
        }
        if (out.words_needed && needed > atoms)
            atomicMax(out.words_needed, needed);
        u64* dst = out.words + position * out.out_words;
        if (needed > atoms)
        {
            for (u32 w = 0; w < out.out_words; ++w)
                dst[w] = 0;
            return;
        }
        for (u32 w = 0; w < atoms; ++w)
            dst[w] = ((w < p.words ? row[w] : 0) & ~dels[w]) | adds[w];
        for (u32 w = 0; w < t.numeric.slots; ++w)
            dst[atoms + w] = values[w];
    }
};

template<u32 OW, bool Write>
__global__ void successors(TaskView t, Parents p, Views v, SchemaSet set, const u32* offsets, u32* counts,
                          Labels labels, SuccessorRows out, u32* error)
{
    const u64 at = u64{blockIdx.x} * blockDim.x + threadIdx.x;
    const u32 si = static_cast<u32>(at % set.count), parent = static_cast<u32>(at / set.count);
    const u32 live = p.rows_dev && *p.rows_dev < p.rows ? *p.rows_dev : p.rows;
    if (parent >= live || (Write && labels.live && !*labels.live))
        return;
    const u32 schema = set.schemas[si], segment = parent * set.num_schemas + schema;
    const u32* sc = t.schema + u64{schema} * k_sc_count;
    const u32* mx = t.matcher + u64{sc[set.witness ? k_sc_pre0 : k_sc_pre1]} * k_mc_count;
    const u64* row = p.data + u64{parent} * p.stride;
    const u64* derived = p.derived ? p.derived + u64{parent} * p.derived_words : nullptr;
    const StateRef state{row, p.words, derived, p.derived_words, row + p.words};
    const u64* view = v.data + u64{parent} * v.words;
    ActionWriter<OW> writer{t, p, v, out, row, derived, schema, parent, error};
    u32 bind[k_max_depth];
    auto run = [&](auto& emit)
    {
        if (set.fc && OW <= k_max_fc_ow)
            run_fc<OW, false, false, true>(t, mx, state, view, bind, emit);
        else
            run_fixed<OW, k_max_depth, false, false, true>(t, mx, state, view, bind, emit);
    };
    u32 count = 0;
    if constexpr (!Write)
    {
        auto emit = [&](const u32* b) { if (writer.collect(b)) ++count; };
        run(emit);
        counts[segment] = count;
    }
    else
    {
        const u32 first = offsets[segment], end = offsets[segment + 1], arity = sc[k_sc_arity];
        const bool sort = arity > 1 && set.canonical && device_sorts(mx[k_mc_flags], set.fc != 0);
        u32* scratch = nullptr;
        if (sort && end > first)
        {
            if (labels.binding && end <= labels.capacity)
                scratch = labels.binding + u64{first} * labels.label_width;
            else if (labels.scratch && labels.scratch_indexed && end <= labels.scratch_rows)
                scratch = labels.scratch + u64{first} * labels.label_width;
            else if (labels.scratch && !labels.scratch_indexed && end - first <= labels.scratch_rows)
                scratch = labels.scratch;
            if (!scratch)
            {
                if (labels.error) atomicOr(labels.error, 1u);
                return;
            }
        }
        auto put = [&](u32 index, const u32* b)
        {
            const u64 pos = u64{first} + index;
            if (pos >= labels.capacity)
            {
                if (out.words_needed)
                {
                    u32 need = 0;
                    for (u32 w = 0; w < k_max_words; ++w)
                        if (((w < p.words ? row[w] : 0) & ~writer.dels[w]) | writer.adds[w]) need = w + 1;
                    if (need > out.out_words - t.numeric.slots)
                        atomicMax(out.words_needed, need);
                }
                return;
            }
            if (labels.schema) labels.schema[pos] = schema;
            if (labels.parent) labels.parent[pos] = labels.parent_base + parent;
            if (labels.binding)
                for (u32 i = 0; i < labels.label_width; ++i)
                    labels.binding[pos * labels.label_width + i] = i < arity ? b[i] : k_none;
            if (out.words) writer.put(pos);
        };
        auto emit = [&](const u32* b)
        {
            if (!writer.collect(b)) return;
            if (sort)
                for (u32 i = 0; i < arity; ++i) scratch[u64{count} * labels.label_width + i] = b[i];
            else
                put(count, b);
            ++count;
        };
        run(emit);
        if (sort)
        {
            sort_rows(scratch, count, labels.label_width, arity);
            for (u32 i = 0; i < count; ++i)
            {
                const u32* b = scratch + u64{i} * labels.label_width;
                writer.collect(b);
                put(i, b);
            }
        }
    }
}

__global__ void goals(TaskView t, Parents p, u32* count, const u32* order, u64 n, u8* flags)
{
    const u64 index = u64{blockIdx.x} * blockDim.x + threadIdx.x;
    if (index >= n) return;
    const u32 live = p.rows_dev && *p.rows_dev < p.rows ? *p.rows_dev : p.rows;
    const u64 row = order ? order[index] : index;
    bool ok = row < live;
    if (ok)
    {
        const u64* state = p.data + row * p.stride;
        const u64* der = p.derived ? p.derived + row * p.derived_words : nullptr;
        ok = goal_holds(t, state, p.words, der, p.derived_words);
        for (u32 i = 0; i < t.numeric.goal_count && ok; ++i)
            ok = numeric_holds(t.numeric, t.numeric.goal_begin + i, state + p.words, nullptr, t.num_objects);
    }
    if (flags) flags[index] = ok;
    if (ok && count)
    {
        atomicAdd(count, 1u);
        atomicMin(count + 1, static_cast<u32>(index));
    }
}

__global__ void convert(TaskView t, const u64* src, u64 ss, u32 sw, u64* dst, u64 ds, u32 dw, u64 n, bool to_device)
{
    const u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u64* x = src + i * ss;
    u64* y = dst + i * ds;
    for (u32 w = 0; w < dw; ++w) y[w] = w < sw ? x[w] : 0;
    const auto& num = t.numeric;
    if (to_device)
        for (u32 k = 0; k < num.slots; ++k)
        {
            const f64 value = num.storage ? static_cast<f64>(static_cast<i32>(static_cast<u32>(x[sw + (k >> 1)] >> ((k & 1) * 32))))
                                          : numeric_value(x + sw, k);
            y[dw + k] = numeric_bits(value);
        }
    else
    {
        const u32 nw = num.storage ? (num.slots + 1) / 2 : num.slots;
        for (u32 k = 0; k < nw; ++k) y[dw + k] = 0;
        for (u32 k = 0; k < num.slots; ++k)
            if (num.storage)
                y[dw + (k >> 1)] |= u64{static_cast<u32>(static_cast<i32>(numeric_value(x + sw, k)))} << ((k & 1) * 32);
            else
                y[dw + k] = x[sw + k];
    }
}

template<bool Write>
cudaError_t launch(TaskView t, Parents p, Views v, SchemaSet set, const u32* offsets, u32* counts,
                   Labels labels, SuccessorRows out, u32* error, cudaStream_t stream)
{
    if (!p.rows || !set.count) return cudaSuccess;
    const u64 n = u64{p.rows} * set.count;
    switch (t.ow)
    {
#define RUN(OW) case OW: successors<OW, Write><<<grid(n), block, 0, stream>>>(t, p, v, set, offsets, counts, labels, out, error); break
        RUN(1); RUN(2); RUN(3); RUN(4); RUN(5); RUN(6); RUN(7); RUN(8);
#undef RUN
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
}  // namespace

cudaError_t launch_count(TaskView t, Parents p, Views v, SchemaSet schemas, u32* counts, u32* error, cudaStream_t stream)
{
    return launch<false>(t, p, v, schemas, nullptr, counts, {}, {}, error, stream);
}
cudaError_t launch_write(TaskView t, Parents p, Views v, SchemaSet schemas, const u32* offsets, Labels labels,
                         SuccessorRows out, u32* error, cudaStream_t stream)
{
    return launch<true>(t, p, v, schemas, offsets, nullptr, labels, out, error, stream);
}
cudaError_t launch_goals(TaskView t, Parents p, u32* count, const u32* order, u64 n, u8* flags, cudaStream_t stream)
{
    if (n) goals<<<grid(n), block, 0, stream>>>(t, p, count, order, n, flags);
    return cudaGetLastError();
}
cudaError_t launch_convert(TaskView t, const u64* src, u64 ss, u32 sw, u64* dst, u64 ds, u32 dw, u64 n,
                           bool to_device, cudaStream_t stream)
{
    if (n) convert<<<grid(n), block, 0, stream>>>(t, src, ss, sw, dst, ds, dw, n, to_device);
    return cudaGetLastError();
}
}  // namespace mymyr::cuda::numeric
