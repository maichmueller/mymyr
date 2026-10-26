#include "mymyr/cuda/runtime.hpp"
#include "mymyr/cuda/translate.hpp"

#include <gtest/gtest.h>
#include <array>
#include <bit>

using namespace mymyr;
namespace
{
TEST(DeviceTranslate, MissingAtomsConstantsAndCapturedReplay)
{
    const auto ctx = cuda::DeviceContext::create(0);
    const auto stream = ctx->stream();
    cuda::TranslateInstance inst;
    inst.slots = 3;
    inst.source_words = 1;
    inst.target_words = 2;
    inst.target_slots = 70;
    const std::array<u32, 3> slots{65, 0, ~u32{0}};
    const std::array<u64, 2> constants{0, 4}, source{3, 4};
    cuda::DeviceBuffer di(ctx, sizeof(inst)), dm(ctx, sizeof(slots)), dc(ctx, sizeof(constants)), ds(ctx, sizeof(source));
    cuda::DeviceBuffer out(ctx, 4 * sizeof(u64)), mask(ctx, 2), status(ctx, 3 * sizeof(i32));
    const auto upload = [&](void* dst, const void* src, usize bytes) {
        cuda::check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream), "translate test upload");
    };
    upload(di.data(), &inst, sizeof(inst));
    upload(dm.data(), slots.data(), sizeof(slots));
    upload(dc.data(), constants.data(), sizeof(constants));
    upload(ds.data(), source.data(), sizeof(source));
    ctx->synchronize();
    const cuda::TranslateView view{static_cast<const cuda::TranslateInstance*>(di.data()), static_cast<const u32*>(dm.data()),
                                   nullptr, static_cast<const u64*>(dc.data()), nullptr, 1};
    const auto launch = [&] {
        cuda::check(cuda::translate_rows(view, static_cast<const u64*>(ds.data()), 2, 1, 1, nullptr,
                                          static_cast<u64*>(out.data()), 2, 2, static_cast<u8*>(mask.data()),
                                          static_cast<i32*>(status.data()), stream), "translate test");
    };
    auto graph = cuda::GraphExec::capture(stream, launch);
    for (u32 replay = 0; replay < 2; ++replay)
    {
        graph.launch(stream);
        std::array<u64, 4> rows{};
        std::array<u8, 2> lossless{};
        std::array<i32, 3> flags{};
        cuda::check(cudaMemcpyAsync(rows.data(), out.data(), sizeof(rows), cudaMemcpyDeviceToHost, stream), "translate test rows");
        cuda::check(cudaMemcpyAsync(lossless.data(), mask.data(), sizeof(lossless), cudaMemcpyDeviceToHost, stream), "translate test mask");
        cuda::check(cudaMemcpyAsync(flags.data(), status.data(), sizeof(flags), cudaMemcpyDeviceToHost, stream), "translate test status");
        ctx->synchronize();
        EXPECT_EQ(rows, (std::array<u64, 4>{1, 6, 0, 4}));
        EXPECT_EQ(lossless, (std::array<u8, 2>{1, 0}));
        EXPECT_EQ(flags, (std::array<i32, 3>{1, 1, 2}));
    }
}

TEST(DeviceTranslate, PackedIntegersConvertToReorderedDoubles)
{
    const auto ctx = cuda::DeviceContext::create(0);
    const auto stream = ctx->stream();
    cuda::TranslateInstance inst;
    inst.source_words = inst.target_words = 1;
    inst.numeric_slots = inst.target_numeric_slots = 2;
    inst.source_i32 = 1;
    const std::array<u32, 2> numeric{1, 0};
    const std::array<u64, 2> source{0, (u64{7} << 32) | 3}, initial{0, 0};
    const u64 constant = 0;
    cuda::DeviceBuffer di(ctx, sizeof(inst)), dn(ctx, sizeof(numeric)), dc(ctx, 8), dv(ctx, sizeof(initial)), ds(ctx, sizeof(source));
    cuda::DeviceBuffer out(ctx, 3 * sizeof(u64)), mask(ctx, 1), status(ctx, 3 * sizeof(i32));
    const auto upload = [&](void* dst, const void* src, usize bytes) {
        cuda::check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream), "translate numeric upload");
    };
    upload(di.data(), &inst, sizeof(inst));
    upload(dn.data(), numeric.data(), sizeof(numeric));
    upload(dc.data(), &constant, 8);
    upload(dv.data(), initial.data(), sizeof(initial));
    upload(ds.data(), source.data(), sizeof(source));
    const cuda::TranslateView view{static_cast<const cuda::TranslateInstance*>(di.data()), nullptr,
                                   static_cast<const u32*>(dn.data()), static_cast<const u64*>(dc.data()),
                                   static_cast<const u64*>(dv.data()), 1};
    cuda::check(cuda::translate_rows(view, static_cast<const u64*>(ds.data()), 1, 2, 1, nullptr,
                                      static_cast<u64*>(out.data()), 1, 3, static_cast<u8*>(mask.data()),
                                      static_cast<i32*>(status.data()), stream), "translate numeric test");
    std::array<u64, 3> rows{};
    std::array<i32, 3> flags{};
    cuda::check(cudaMemcpyAsync(rows.data(), out.data(), sizeof(rows), cudaMemcpyDeviceToHost, stream), "translate numeric rows");
    cuda::check(cudaMemcpyAsync(flags.data(), status.data(), sizeof(flags), cudaMemcpyDeviceToHost, stream), "translate numeric status");
    ctx->synchronize();
    EXPECT_DOUBLE_EQ(std::bit_cast<f64>(rows[1]), 7);
    EXPECT_DOUBLE_EQ(std::bit_cast<f64>(rows[2]), 3);
    EXPECT_EQ(flags, (std::array<i32, 3>{0, -1, -1}));
}
}
