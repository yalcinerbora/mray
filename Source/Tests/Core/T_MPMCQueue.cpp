#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <random>
#include <numeric>

#include "Core/MPMCQueue.h"
#include "Core/Vector.h"

#include "GTestWrappers.h"


TEST(MPMCQueueTest, Stress)
{
    static constexpr uint32_t StressIterations = 8;
    static constexpr uint32_t MaxQueueSize = 2;
    static constexpr uint32_t MaxPushSize = 20;
    std::mt19937 rng(123);

    for(uint32_t iter = 0; iter < StressIterations; iter++)
    {
        std::uniform_int_distribution<uint32_t> threadDist(1, std::thread::hardware_concurrency());
        std::uniform_int_distribution<uint32_t> queueDist(1, MaxQueueSize);
        std::uniform_int_distribution<uint32_t> itemDist(1, MaxPushSize);

        uint32_t producerThreadCount = threadDist(rng);
        uint32_t consumerThreadCount = threadDist(rng);
        uint32_t itemPerThreadCount = itemDist(rng);
        uint32_t queueSize = queueDist(rng);

        MPMCQueue<uint32_t> queue(queueSize);
        std::vector<std::thread> producerThreads;
        std::vector<std::thread> consumerThreads;

        for(uint32_t i = 0; i < producerThreadCount; i++)
        {
            producerThreads.emplace_back([&]()
            {
                for(uint32_t i = 0; i < itemPerThreadCount; i++)
                    queue.Enqueue(uint32_t(i + 1));
            });
        }

        std::vector<uint32_t> localReductions(consumerThreadCount, 0u);
        for(uint32_t i = 0; i < consumerThreadCount; i++)
        {
            consumerThreads.emplace_back([&](uint32_t index)
            {
                while(!queue.IsTerminated())
                {
                    uint32_t v;
                    queue.Dequeue(v);
                    localReductions[index] += v;
                }
            }, i);
        }

        // MPMCQueue is not a thread pool so it does not have "future"
        // Wait a little here these operations should finish in a second
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(200ms);

        uint32_t total = std::reduce(localReductions.cbegin(), localReductions.cend(), 0u);

        uint32_t expected = producerThreadCount * (itemPerThreadCount * (itemPerThreadCount + 1)) / 2;

        EXPECT_EQ(expected, total);

        // Gracefully terminate
        queue.Terminate();
        for(std::thread& t : producerThreads)
            t.join();
        for(std::thread& t : consumerThreads)
            t.join();
    }
}