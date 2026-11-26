#include "../spu_coh_bitmap.h"

#include <array>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <thread>

extern "C" int spu_coh_bitmap_concurrent_test(void)
{
    constexpr std::size_t thread_count = 8;
    constexpr unsigned rounds = 4096;
    constexpr unsigned char all_bits = 0xFF;

    unsigned char byte = 0;
    std::atomic<int> failed{0};
    std::barrier<> gate(static_cast<std::ptrdiff_t>(thread_count));
    std::array<std::thread, thread_count> workers;

    for (std::size_t id = 0; id < thread_count; ++id) {
        workers[id] = std::thread([&, id] {
            const auto bit = static_cast<unsigned char>(1u << id);
            for (unsigned pass = 0; pass < rounds; ++pass) {
                if (id == 0)
                    __atomic_store_n(&byte, 0, __ATOMIC_RELAXED);
                gate.arrive_and_wait();
                spu_coh_bitmap_mark_bit(&byte, bit);
                gate.arrive_and_wait();
                if (id == 0 && __atomic_load_n(&byte, __ATOMIC_RELAXED) != all_bits)
                    failed.store(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& worker : workers)
        worker.join();
    return failed.load(std::memory_order_relaxed);
}
