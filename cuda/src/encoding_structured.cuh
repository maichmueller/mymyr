// Serial graph operations share the host's canonical merges while independent graphs run in parallel: a thread per
// graph (EncodeParams::structured_lanes 1), or one lane of every 4, 8 or 32 when the shared memory of a block holds the
// predicate scratch of fewer graphs.
// Included inside encoding.cu's implementation namespace, after its scan and output helpers.

__device__ bool structured_row(const EncodeParams& p, u64 g, RowView& r, ed::HorizonGraph& h)
{
    if (!row_of<true, true>(p, g, r))
        return false;
    if (!p.horizon_mode && !p.actions.offsets)
        return true;
    if (!p.horizon_mode)
    {
        const auto& a = p.actions;
        const i32 first = a.offsets[g], last = a.offsets[g + 1];
        if (a.offsets[0] != 0 || first < 0 || last < first || u64(last) > a.capacity ||
            (a.overflow && *a.overflow))
            return false;
        r.in.actions = {a.schema ? a.schema + first : nullptr,
                        a.binding ? a.binding + u64(first) * a.label_width : a.schema,
                        a.parent ? a.parent + first : nullptr, a.label_width, u32(last - first), g,
                        p.action_storage + g * p.action_stride,
                        p.action_storage + g * p.action_stride + r.v.num_schemas + 1,
                        a.index ? a.index + first : nullptr};
        return true;
    }
    const auto& hz = p.horizon;
    const i32 n0 = hz.node_offsets[g], n1 = hz.node_offsets[g + 1];
    const i32 e0 = hz.edge_offsets[g], e1 = hz.edge_offsets[g + 1];
    if (hz.node_offsets[0] != 0 || hz.edge_offsets[0] != 0 || n0 < 0 || n1 <= n0 || u64(n1) > hz.nodes ||
        e0 < 0 || e1 < e0 || u64(e1) > hz.edges)
        return false;
    r.in.atoms = p.states + u64(n0) * p.state_stride;
    r.in.derived = p.derived ? p.derived + u64(n0) * p.derived_stride : nullptr;
    h = {};
    h.v = &r.v;
    h.row = r.in;
    h.goal = ed::goal_words(r.v, h.row);
    h.nodes = r.in.atoms;
    h.node_stride = p.state_stride;
    h.derived = r.in.derived;
    h.derived_stride = p.derived_stride;
    h.count = u32(n1 - n0);
    h.edges = u32(e1 - e0);
    h.src = hz.src ? hz.src + e0 : nullptr;
    h.dst = hz.dst ? hz.dst + e0 : nullptr;
    h.schema = hz.schema ? hz.schema + e0 : nullptr;
    h.binding = hz.binding ? hz.binding + u64(e0) * hz.label_width : nullptr;
    h.label_width = hz.label_width;
    h.schema_relation = p.schema_relations;
    h.order = hz.scratch + 3 * g + 3 * u64(n0) + 3 * u64(e0);
    const u64 order_words = 3 * p.graphs + 3 * hz.nodes + 3 * hz.edges;
    h.kin = p.horizon_kin ? hz.scratch + order_words + 3 * g + 5 * u64(n0) + 2 * u64(e0) : nullptr;
    h.use_actions = p.horizon_actions;
    h.one_schema_per_relation = false;
    h.use_parents = p.horizon_parents;
    h.use_cousins = p.horizon_cousins;
    h.exclude = p.horizon_exclude;
    h.root_target = p.horizon_root_target;
    if (h.src)
        for (u32 e = 0; e < h.edges; ++e)
        {
            if (u32(h.src[e]) >= h.count || !h.dst || u32(h.dst[e]) >= h.count)
                return false;
            if (!h.use_actions)
                continue;
            const i32 s = h.schema[e];
            if (s < 0 || u32(s) >= r.v.num_schemas || h.schema_relation[s] < 0 ||
                r.v.schema_arity[s] > h.label_width)
                return false;
            for (u32 k = 0; k < r.v.schema_arity[s]; ++k)
                if (u32(h.binding[u64(e) * h.label_width + k]) >= r.v.num_objects)
                    return false;
        }
    for (u32 n = 0; n < h.count; ++n)
        if (ed::has_foreign_atom_bits(r.v, ed::horizon_atoms(h, n)) ||
            (h.derived && ed::has_foreign_derived_bits(r.v, ed::horizon_derived(h, n))))
            return false;
    if (ed::has_foreign_goal_bits(r.v, r.in) ||
        (r.in.layers && ed::has_foreign_layer_bits(r.v, r.in.layers->subgoals)))
        return false;
    return true;
}

__global__ void structured_count_kernel(const EncodeParams params)
{
    const EncodeParams p = live_params(params);
    extern __shared__ __align__(8) unsigned char shared_bytes[];
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        if ((p.actions.offsets && p.actions.offsets[0] != 0) ||
            (p.horizon_mode && (p.horizon.node_offsets[0] != 0 || p.horizon.edge_offsets[0] != 0)))
            atomicOr(p.flags, 1u);
    }
    const u32 lanes = p.structured_lanes;
    if (threadIdx.x % lanes)
        return;
    const u64 gi = (u64(blockIdx.x) * k_block + threadIdx.x) / lanes;
    if (gi >= p.graphs)
        return;
    const u64 g = p.order ? p.order[gi] : gi;
    if (g >= p.graphs)
    {
        atomicOr(p.flags, 1u);
        return;
    }
    u32* rank = reinterpret_cast<u32*>(shared_bytes) + (threadIdx.x / lanes) * p.context.num_virtual;
    RowView r;
    ed::HorizonGraph h;
    ed::GraphTotals t = ed::empty_totals();
    bool ok = structured_row(p, g, r, h);
    u32* counts = p.cell_counts + g * p.cell_stride;
    if (ok)
    {
        if (p.horizon_mode)
        {
            ed::canonical_horizon_edges(h);
            if (h.kin)
            {
                auto prepared = h;
                prepared.use_cousins = false;  // Cousin rows are gathered into linear scratch on demand.
                ed::horizon_kin_prepare(prepared);
            }
            const ed::CellList list = r.in.goal_pos ? ed::CellList{r.v.row_cells, r.v.num_row_cells}
                                                   : ed::CellList{r.v.cells, r.v.num_cells};
            auto sink = [&](u32 i, const ed::Cell& c, u32 n)
            {
                counts[i] = n;
                if (n == 0xffffffffu)
                    ok = false;
                if (p.relation_major)
                    Totals{p.totals, p.wide}.set(p.segments + 1 + ed::cell_segment(true, g, c.relation,
                                                                                 p.graphs, r.v.num_relations), n);
            };
            if (ok)
                t = ed::horizon_count_graph(r.v, h, list, rank, sink);
        }
        else
        {
            count_levels(ed::SoloGroup{}, p, r, g);
            t = ed::count_graph<true>(r.v, r.in, counts, rank);
            if (p.relation_major)
            {
                const ed::CellList list = ed::cell_list<true>(r.v, r.in);
                for (u32 i = 0; i < list.count; ++i)
                    Totals{p.totals, p.wide}.set(p.segments + 1 + ed::cell_segment(true, g, list.cells[i].relation,
                                                                                 p.graphs, r.v.num_relations), counts[i]);
            }
            ok = (t.flags & ~ed::k_graph_auxiliary) == 0;
        }
    }
    if (!ok)
    {
        atomicOr(p.flags, 1u);
        t = ed::empty_totals();
    }
    const Totals totals{p.totals, p.wide};
    const u64 row = p.graphs + 1;
    totals.set(1 + g, t.entities);
    totals.set(row + 1 + g, t.targets);
    if (!p.relation_major)
    {
        totals.set(2 * row + 1 + g, t.tuples);
        totals.set(3 * row + 1 + g, t.args);
    }
    if (p.history_at)
        totals.set(p.history_at + 1 + g, t.history_steps);
    if (p.numeric_at)
        totals.set(p.numeric_at + 1 + g, ok ? ed::numeric_slot_count(r.v) : 0);
}

template<class I>
__global__ void structured_fill_kernel(const EncodeParams params)
{
    const EncodeParams p = live_params(params);
    extern __shared__ __align__(8) unsigned char shared_bytes[];
    const Totals totals{p.totals, p.wide};
    u64 base = 0;
    auto* first = reinterpret_cast<u64*>(shared_bytes);
    const enc::FlatSizes sizes = p.relation_major ? batch_sizes<true, false>(p, totals, first, base)
                                                 : batch_sizes<false, false>(p, totals, first, base);
    if (!blockIdx.x && !threadIdx.x)
    {
        *p.sizes = sizes;
        if (p.output_graphs)
            *p.output_graphs = p.graphs;
        if (p.numeric_size)
            *p.numeric_size = p.numeric_at ? totals.get(p.numeric_at + p.graphs) - totals.get(p.numeric_at) : 0;
    }
    if (p.graphs_device)
        zero_unused_live<I>(p, sizes, u64(gridDim.x) * blockDim.x);
    else if (p.zero_unused)
        zero_unused<I>(p, sizes);
    if (sizes.overflow)
        return;
    const ed::GraphOut<I> out = graph_out<I>(p);
    auto* entity_offsets = static_cast<I*>(p.out.entity_offsets);
    const u64 G = p.graphs, R = p.context.num_relations, row = G + 1;
    ScanPositions positions{totals, p.segments, base, first, p.context.relation_arity};
    if (!blockIdx.x && !threadIdx.x)
    {
        entity_offsets[G] = I(sizes.entities);
        if (out.target_offsets)
            out.target_offsets[G] = I(sizes.targets);
        if (out.history_offsets)
            out.history_offsets[G] = I(sizes.history);
        out.relation_offsets[G * R] = I(sizes.tuples);
        out.relation_arg_offsets[G * R] = I(sizes.args);
        if (p.simple)
        {
            static_cast<I*>(p.out.node_offsets)[G] = I(sizes.entities + sizes.tuples);
            static_cast<I*>(p.out.edge_offsets)[G] = I(2 * sizes.args);
        }
    }
    if (p.relation_major)
        for (u64 s = u64(blockIdx.x) * k_block + threadIdx.x; s < G * R; s += u64(gridDim.x) * k_block)
        {
            out.relation_offsets[s] = I(positions.tuples_before(s));
            out.relation_arg_offsets[s] = I(positions.args(s, u32(s / G)));
        }
    const u32 lanes = p.structured_lanes;
    if (threadIdx.x % lanes)
        return;
    const u64 gi = (u64(blockIdx.x) * k_block + threadIdx.x) / lanes;
    if (gi >= G)
        return;
    const u64 g = p.order ? p.order[gi] : gi;
    RowView r;
    ed::HorizonGraph h;
    if (!structured_row(p, g, r, h))
        return;
    u32* rank = reinterpret_cast<u32*>(shared_bytes + (p.relation_major ? align8(R * 8) : 0)) +
                (threadIdx.x / lanes) * p.context.num_virtual;
    const u32* counts = p.cell_counts + g * p.cell_stride;
    const u64 entity = totals.get(g), target = totals.get(row + g) - totals.get(row);
    p.out.task_ids[g] = i32(r.instance);
    entity_offsets[g] = I(entity);
    const u64 tuple = p.relation_major ? 0 : totals.get(2 * row + g) - totals.get(2 * row);
    const u64 arg = p.relation_major ? 0 : totals.get(3 * row + g) - totals.get(3 * row);
    if (!p.horizon_mode)
    {
        fill_levels(p, r, g);
        const ed::GraphTotals t = ed::totals_from_counts<true>(r.v, r.in, [&](u32 i, const ed::Cell&) { return counts[i]; });
        const u64 history = p.history_at ? totals.get(p.history_at + g) - totals.get(p.history_at) : 0;
        const u64 numeric = p.numeric_at ? totals.get(p.numeric_at + g) - totals.get(p.numeric_at) : 0;
        if (p.relation_major)
            ed::fill_graph_positioned<I, ScanPositions, true>(r.v, r.in, g, entity, t, out, positions,
                                                             target, rank, counts, history);
        else
            ed::fill_graph<I, true>(r.v, r.in, g, tuple, arg, entity, t, out, target, rank, counts, history, numeric);
        if (p.simple)
        {
            if (p.relation_major)
                fill_simple<I, true, true>(ed::SoloGroup{}, p, r, g, t, out, positions, entity, tuple, arg, counts);
            else
                fill_simple<I, false, true>(ed::SoloGroup{}, p, r, g, t, out, positions, entity, tuple, arg, counts);
        }
        return;
    }
    const ed::CellList list = r.in.goal_pos ? ed::CellList{r.v.row_cells, r.v.num_row_cells}
                                          : ed::CellList{r.v.cells, r.v.num_cells};
    auto sink = [](u32, const ed::Cell&, u32) {};
    const ed::GraphTotals t = ed::horizon_count_graph(r.v, h, list, rank, sink);
    const ed::HorizonEntities entities = ed::horizon_entities(r.v, h, t, entity);
    ed::fill_horizon_entities<I>(r.v, h, entities, g, target, out);
    if (r.v.num_virtual)
    {
        ed::horizon_virtual_ranks(r.v, list, counts, rank);
        ed::fill_virtual_entities<I>(r.v, ed::EntityBlocks{0, 0, 0, 0, 0, entities.virtuals}, rank, out);
    }
    u64 tuples = tuple, args = arg;
    u32 cell = 0;
    for (u32 relation = 0; relation < R; ++relation)
    {
        const u64 segment = ed::cell_segment(p.relation_major, g, relation, G, u32(R));
        if (!p.relation_major)
        {
            out.relation_offsets[segment] = I(tuples);
            out.relation_arg_offsets[segment] = I(args);
        }
        u32 n = 0;
        if (cell < list.count && list.cells[cell].relation == relation)
        {
            const ed::Cell& c = list.cells[cell++];
            u64 predicate = ed::k_no_entity;
            if (r.v.num_virtual)
            {
                const u32 q = r.v.relation_virtual[relation];
                if (q != ed::k_no_virtual && rank[q] != ed::k_no_virtual)
                    predicate = entities.virtuals + rank[q];
            }
            const u64 at = p.relation_major ? positions.args(segment, relation) : args;
            n = ed::fill_horizon_cell<I>(c, h, entities, predicate, out.relation_args + at);
            tuples += n;
            args += u64(n) * c.arity;
        }
        if (out.relation_counts)
            out.relation_counts[g * R + relation] = I(n);
    }
}

__global__ void horizon_instances_kernel(const EncodeParams p, i32* ids)
{
    const u64 g = u64(blockIdx.x) * k_block + threadIdx.x;
    if (g == 0)
    {
        const i32 count = p.horizon.node_offsets[p.graphs];
        const bool valid = count >= 0 && u64(count) <= p.horizon.nodes;
        *reinterpret_cast<u64*>(p.action_storage) = valid ? u64(count) : 0;
        if (!valid)
            atomicOr(p.flags, 1u);
    }
    if (g >= p.graphs)
        return;
    const i32 first = p.horizon.node_offsets[g], last = p.horizon.node_offsets[g + 1];
    if (first < 0 || last <= first || u64(last) > p.horizon.nodes)
        return;
    const i32 id = p.task_ids ? p.task_ids[g] : 0;
    for (i32 n = first; n < last; ++n)
        ids[n] = id;
}
