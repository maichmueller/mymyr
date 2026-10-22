// Traversal uses the state space's unique ids; node rows are copied without comparing or hashing their words.

__device__ bool space_edge_before(const DeviceSpaceHorizonBuild& out, u64 a, u64 b, u32 width)
{
    if (out.dst[a] != out.dst[b])
        return out.dst[a] < out.dst[b];
    if (out.schema[a] != out.schema[b])
        return out.schema[a] < out.schema[b];
    for (u32 k = 0; k < width; ++k)
        if (out.binding[a * width + k] != out.binding[b * width + k])
            return out.binding[a * width + k] < out.binding[b * width + k];
    return out.src[a] < out.src[b];
}

__device__ void space_edge_swap(const DeviceSpaceHorizonBuild& out, u64 a, u64 b, u32 width)
{
    auto swap = [](i32& x, i32& y) { const i32 t = x; x = y; y = t; };
    swap(out.src[a], out.src[b]);
    swap(out.dst[a], out.dst[b]);
    swap(out.schema[a], out.schema[b]);
    for (u32 k = 0; k < width; ++k)
        swap(out.binding[a * width + k], out.binding[b * width + k]);
}

__device__ void space_edge_sift(const DeviceSpaceHorizonBuild& out, u64 base, u32 count, u32 root, u32 width)
{
    while (u64(root) * 2 + 1 < count)
    {
        u32 child = 2 * root + 1;
        if (child + 1 < count && space_edge_before(out, base + child, base + child + 1, width))
            ++child;
        if (!space_edge_before(out, base + root, base + child, width))
            break;
        space_edge_swap(out, base + root, base + child, width);
        root = child;
    }
}

__global__ void space_horizon_kernel(const DeviceHorizonSpace space, const DeviceSpaceHorizonBuild out)
{
    out.node_offsets[0] = out.edge_offsets[0] = 0;
    if (!out.graphs)
        return;
    const u32 limit = out.limit ? min(out.limit, space.states) : space.states;
    i32* const map = out.scratch;
    i32* const queue = out.scratch + space.states;
    u64 nodes = 0, edges = 0;
    for (u32 g = 0; g < out.graphs; ++g)
    {
        bool valid = u32(out.roots[g]) < space.states;
        bool truncated = false;
        for (u32 n = 0; n < space.states; ++n)
            map[n] = -1;
        u32 count = valid ? 1 : 0, first = 0, last = count;
        if (valid)
        {
            queue[0] = out.roots[g];
            map[u32(queue[0])] = 0;
        }
        const u64 edge_base = edges;
        for (u32 level = 0; valid && level < out.depth && first < last; ++level)
        {
            for (u32 src = first; valid && src < last; ++src)
            {
                const u32 state = u32(queue[src]);
                const u64 begin = space.offsets[state], end = space.offsets[state + 1];
                if (begin > end || end > space.edges)
                {
                    valid = false;
                    break;
                }
                for (u64 e = begin; e < end; ++e)
                {
                    const u32 target = space.targets[e];
                    if (target >= space.states)
                    {
                        valid = false;
                        break;
                    }
                    i32 dst = map[target];
                    if (dst < 0)
                    {
                        if (count == limit)
                        {
                            truncated = true;
                            if (!out.truncate)
                                valid = false;
                            if (!valid)
                                break;
                            continue;
                        }
                        dst = static_cast<i32>(count);
                        map[target] = dst;
                        queue[count++] = static_cast<i32>(target);
                    }
                    out.src[edges] = static_cast<i32>(src);
                    out.dst[edges] = dst;
                    out.schema[edges] = static_cast<i32>(space.schemas[e]);
                    for (u32 k = 0; k < space.label_width; ++k)
                        out.binding[edges * space.label_width + k] = static_cast<i32>(space.bindings[e * space.label_width + k]);
                    ++edges;
                }
            }
            first = last;
            last = count;
        }
        if (!valid)
        {
            for (u32 i = g + 1; i <= out.graphs; ++i)
                out.node_offsets[i] = out.edge_offsets[i] = -1;
            return;
        }
        for (u32 n = 0; n < count; ++n)
            for (u32 w = 0; w < space.words; ++w)
                out.nodes[(nodes + n) * space.words + w] = space.rows[u64(queue[n]) * space.words + w];
        const u32 edge_count = static_cast<u32>(edges - edge_base);
        for (u32 i = edge_count / 2; i > 0; --i)
            space_edge_sift(out, edge_base, edge_count, i - 1, space.label_width);
        for (u32 end = edge_count; end > 1; --end)
        {
            space_edge_swap(out, edge_base, edge_base + end - 1, space.label_width);
            space_edge_sift(out, edge_base, end - 1, 0, space.label_width);
        }
        nodes += count;
        out.node_offsets[g + 1] = static_cast<i32>(nodes);
        out.edge_offsets[g + 1] = static_cast<i32>(edges);
        if (out.truncated)
            out.truncated[g] = truncated;
    }
}
