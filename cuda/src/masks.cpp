// Lifted action masks on the device (include/mymyr/cuda/masks.hpp): the host driver of cuda/mask_kernels.hpp.

#include "mymyr/cuda/masks.hpp"

#include "mymyr/cuda/generator.hpp"
#include "mymyr/cuda/lifted.hpp"
#include "mymyr/cuda/mask_kernels.hpp"
#include "table_launch.hpp"

#include <algorithm>
#include <stdexcept>

namespace mymyr::cuda
{
namespace
{
std::string task_why(const Task& task)
{
    if (std::string why = ChunkGenerator::unsupported(task); !why.empty())
        return why;
    if (task.numeric_slots())
        return "numeric fluents (their numeric effect rules run on the host)";
    if (task.has_axioms())
        return "axioms (derived atoms are evaluated on the host)";
    for (const plan::Schema& ps : task.compiled().schemas)
        if (ps.arity > lifted::k_max_depth)
            return "a schema of " + std::to_string(ps.arity) + " parameters (the device masks take at most " +
                   std::to_string(lifted::k_max_depth) + ")";
    return {};
}

template<class T>
DeviceBuffer upload(const ContextPtr& ctx, const std::vector<T>& v, cudaStream_t s)
{
    DeviceBuffer b(ctx, std::max<u64>(v.size(), 1) * sizeof(T), s);
    if (!v.empty())
        check(cudaMemcpyAsync(b.data(), v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, s), "cudaMemcpyAsync");
    return b;
}
}  // namespace

std::string DeviceMasks::unsupported(const rl::TaskSuite& suite)
{
    for (u32 g = 0; g < suite.size(); ++g)
        if (const std::string why = task_why(*suite.task(g)); !why.empty())
            return suite.size() > 1 ? "instance " + std::to_string(g) + ": " + why : why;
    return {};
}

DeviceMasks::DeviceMasks(ContextPtr ctx, rl::TaskSuitePtr suite) : m_ctx(std::move(ctx)), m_suite(std::move(suite))
{
    if (!m_ctx || !m_suite)
        throw std::invalid_argument("mymyr: DeviceMasks: null context or suite");
    if (const std::string why = unsupported(*m_suite); !why.empty())
        throw std::invalid_argument("mymyr: the device masks cannot run this " + std::string(m_suite->noun()) + ": " +
                                    why);
    DeviceGuard g(m_ctx->device());
    const cudaStream_t s = m_ctx->stream();
    for (u32 d = 0; d < m_suite->num_domains(); ++d)
    {
        m_tables.push_back(DeviceTaskTable::upload(m_ctx, m_suite->table(d)));
        m_tables.back()->acquire(s);
    }
    const u32 I = m_suite->size();
    std::vector<rl::dev::TaskView> views(I);
    std::vector<u32> words(I), slots(I);
    for (u32 k = 0; k < I; ++k)
    {
        const rl::TaskSuite::Ref r = m_suite->ref(k);
        views[k] = m_tables[r.domain]->view(r.local);
        const Task& task = *m_suite->task(k);
        words[k] = task.words();
        slots[k] = task.atoms().fluent_slots();
        m_view_words = std::max<u64>(m_view_words, u64{views[k].view_rows} * views[k].ow);
        if (std::find(m_ows.begin(), m_ows.end(), views[k].ow) == m_ows.end())
            m_ows.push_back(views[k].ow);
    }
    m_view0 = views.at(0);
    m_views = upload(m_ctx, views, s);
    m_words = upload(m_ctx, words, s);
    m_slots = upload(m_ctx, slots, s);
    // the uploads are complete before any call: the calls need no wait (and may be captured)
    check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
}

const rl::dev::TaskView* DeviceMasks::device_views() const noexcept
{
    return static_cast<const rl::dev::TaskView*>(m_views.data());
}

void DeviceMasks::use(cudaStream_t s) const
{
    // the buffers' frees wait for the work of every stream they were used on
    std::lock_guard lock(m_mutex);
    if (std::find(m_streams.begin(), m_streams.end(), s) != m_streams.end())
        return;
    if (!detail::capturing(s))
        for (const DeviceTaskTablePtr& t : m_tables)
            t->acquire(s);
    for (const DeviceBuffer* b : {&m_views, &m_words, &m_slots})
        const_cast<DeviceBuffer*>(b)->record_stream(s);
    m_streams.push_back(s);
}

void DeviceMasks::views_of(const Batch& b, u64* views, cudaStream_t s) const
{
    if (b.rows > 0xFFFFFFFFull)
        throw std::invalid_argument("mymyr: DeviceMasks: more than 2^32 - 1 rows");
    const lifted::Parents p{b.states, b.stride, b.words, static_cast<u32>(b.rows), nullptr, 0};
    const lifted::Views v{views, m_view_words};
    if (!b.task_ids)
    {
        check(lifted::launch_view(m_view0, p, v, s), "launch_view");
        return;
    }
    lifted::Multi m;
    m.views = device_views();
    m.inst = reinterpret_cast<const u32*>(b.task_ids);
    m.instances = m_suite->size();
    check(lifted::launch_view_multi(m, p, v, s), "launch_view_multi");
}

void DeviceMasks::check_rows(const Batch& b, u32* status, cudaStream_t s) const
{
    const maskk::Rows r{b.states, b.stride, b.words, b.rows, b.task_ids, device_views(), m_suite->size()};
    check(maskk::launch_check(r, static_cast<const u32*>(m_words.data()), static_cast<const u32*>(m_slots.data()),
                                    status, s),
                "launch_check");
}

namespace
{
void validate(const DeviceMasks& x, const DeviceMasks::Batch& b, const void* out, const u32* status, const char* what)
{
    if (b.rows && (!b.states || !out || !status))
        throw std::invalid_argument(std::string("mymyr: ") + what + ": null rows, output or status");
    if (b.stride < b.words)
        throw std::invalid_argument(std::string("mymyr: ") + what + ": row stride smaller than the row width");
    if (b.rows && !b.task_ids && x.suite()->size() > 1)
        throw std::invalid_argument(std::string("mymyr: ") + what + ": a batch of several instances needs task ids");
}
}  // namespace

void DeviceMasks::schemas(const Batch& b, u8* out, u64* views, u32* status, cudaStream_t s) const
{
    validate(*this, b, out, status, "schema masks");
    DeviceGuard g(m_ctx->device());
    use(s);
    const u32 S = m_suite->max_schemas();
    cuda::check(cudaMemsetAsync(status, 0, sizeof(u32), s), "cudaMemsetAsync");
    if (b.rows == 0)
        return;
    if (S)
        cuda::check(cudaMemsetAsync(out, 0, b.rows * S, s), "cudaMemsetAsync");
    check_rows(b, status, s);
    views_of(b, views, s);
    const maskk::Rows r{b.states, b.stride, b.words, b.rows, b.task_ids, device_views(), m_suite->size()};
    for (const u32 ow : m_ows)
        cuda::check(maskk::launch_schema_masks(r, {views, m_view_words}, ow, S, out, s), "launch_schema_masks");
}

void DeviceMasks::objects(const Batch& b, Schemas schema, const i32* prefix, u32 stride, u32 depth, u8* out,
                          u32 num_objects, u64* views, u32* status, cudaStream_t s) const
{
    validate(*this, b, out, status, "object masks");
    if (num_objects < m_suite->max_objects())
        throw std::invalid_argument("mymyr: object masks: num_objects is below the suite's maximum");
    if (stride < depth || (b.rows && depth && !prefix))
        throw std::invalid_argument("mymyr: object masks: prefix needs at least depth columns");
    DeviceGuard g(m_ctx->device());
    use(s);
    cuda::check(cudaMemsetAsync(status, 0, sizeof(u32), s), "cudaMemsetAsync");
    if (b.rows == 0)
        return;
    if (num_objects)
        cuda::check(cudaMemsetAsync(out, 0, b.rows * num_objects, s), "cudaMemsetAsync");
    check_rows(b, status, s);
    views_of(b, views, s);
    const maskk::Rows r{b.states, b.stride, b.words, b.rows, b.task_ids, device_views(), m_suite->size()};
    const maskk::ObjectMasks m{{schema.schema, schema.value}, prefix, stride, depth, out, num_objects};
    for (const u32 ow : m_ows)
        cuda::check(maskk::launch_object_masks(r, {views, m_view_words}, ow, m, status, s), "launch_object_masks");
}

void DeviceMasks::applicable(const Batch& b, Schemas schema, const i32* binding, u32 width, u8* out, u32* status,
                             cudaStream_t s) const
{
    validate(*this, b, out, status, "is_applicable");
    if (b.rows && width && !binding)
        throw std::invalid_argument("mymyr: is_applicable: null binding");
    DeviceGuard g(m_ctx->device());
    use(s);
    cuda::check(cudaMemsetAsync(status, 0, sizeof(u32), s), "cudaMemsetAsync");
    if (b.rows == 0)
        return;
    check_rows(b, status, s);
    const maskk::Rows r{b.states, b.stride, b.words, b.rows, b.task_ids, device_views(), m_suite->size()};
    cuda::check(maskk::launch_applicable(r, {schema.schema, schema.value}, binding, width, out, status, s),
                "launch_applicable");
}

void DeviceMasks::any(const Batch& b, const u8* skip, u32* out, u64 out_stride, u64* views, cudaStream_t s) const
{
    if (b.rows == 0)
        return;
    if (!b.states || !out || !views)
        throw std::invalid_argument("mymyr: DeviceMasks::any: null rows, output or views");
    DeviceGuard g(m_ctx->device());
    use(s);
    views_of(b, views, s);
    cuda::check(maskk::launch_any_clear(b.rows, skip, out, out_stride, s), "launch_any_clear");
    const maskk::Rows r{b.states, b.stride, b.words, b.rows, b.task_ids, device_views(), m_suite->size()};
    for (const u32 ow : m_ows)
        cuda::check(maskk::launch_any(r, {views, m_view_words}, ow, m_suite->max_schemas(), skip, out, out_stride, s),
                    "launch_any");
}
}  // namespace mymyr::cuda
