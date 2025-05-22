#include <sstream>

#include <gtest/gtest.h>

#include "SceneLoaderMRay/EntryPoint.h"

#include "Core/Types.h"
#include "Core/SharedLibrary.h"
#include "Core/SceneLoaderI.h"
#include "Core/ThreadPool.h"

#include "MockTracer.h"
#include "TestScenes.h"

class SceneLoaderMRayTest : public ::testing::Test
{
    protected:
    // Dunno where or when the destructor is called
    // So everything is wrapped on unique_ptrs
    std::unique_ptr<ThreadPool>         pool = nullptr;
    std::unique_ptr<SharedLibrary>      dllFile = nullptr;
    SharedLibPtr<SceneLoaderI>          loader = {nullptr, nullptr};

    void SetUp() override;
    void TearDown() override;
};

void SceneLoaderMRayTest::SetUp()
{
    unsigned int tCount = std::max(1u, std::thread::hardware_concurrency());
    pool = std::make_unique<ThreadPool>(tCount);
    dllFile = std::make_unique<SharedLibrary>("SceneLoaderMRay");

    SharedLibArgs args
    {
        .mangledConstructorName = "ConstructSceneLoaderMRay",
        .mangledDestructorName = "DestroySceneLoaderMRay"
    };
    MRayError e = dllFile->GenerateObjectWithArgs<SceneLoaderConstructorArgs>(loader, args,
                                                                                     *pool);
    EXPECT_TRUE(!static_cast<bool>(e));
}

void SceneLoaderMRayTest::TearDown()
{
    pool = nullptr;
    loader = {nullptr, nullptr};
    dllFile = nullptr;
}

TEST_F(SceneLoaderMRayTest, Empty)
{
    TracerMock tracer;
    std::istringstream ss{std::string(EmptyScene)};
    auto result = loader->LoadScene(tracer, ss);
    EXPECT_TRUE(result.has_error());
}

TEST_F(SceneLoaderMRayTest, MinimalValid)
{
    TracerMock tracer;
    std::istringstream ss{std::string(MinimalValidScene)};
    auto result = loader->LoadScene(tracer, ss);
    EXPECT_FALSE(result.has_error());

    // Might as well print the error here
    if(result.has_error())
        MRAY_ERROR_LOG("{}", result.error().GetError());
}

TEST_F(SceneLoaderMRayTest, Basic)
{
    TracerMock tracer;
    std::istringstream ss{std::string(BasicScene)};
    auto result = loader->LoadScene(tracer, ss);
    EXPECT_FALSE(result.has_error());

    if(result.has_error())
        MRAY_ERROR_LOG("{}", result.error().GetError());
}

TEST_F(SceneLoaderMRayTest, Kitchen)
{
    static constexpr size_t TOTAL_RUNS = 16;
    double all = 0.0;
    for(uint32_t i = 0; i < TOTAL_RUNS; i++)
    {
        TracerMock tracer(false);
        auto result = loader->LoadScene(tracer, "Scenes/Kitchen/Kitchen.json");
        EXPECT_FALSE(result.has_error());

        if(result.has_error())
        {
            MRAY_LOG("Err! :: {}", result.error().GetError());
            ASSERT_FALSE(true);
        }
        else all += result.value().loadTimeMS;

        loader->ClearScene();
    }
    MRAY_LOG("Average: {:.3f}ms", all / static_cast<double>(TOTAL_RUNS));
}

TEST_F(SceneLoaderMRayTest, KitchenGFG)
{

    static constexpr size_t TOTAL_RUNS = 16;
    double all = 0.0;
    for(uint32_t i = 0; i < TOTAL_RUNS; i++)
    {
        TracerMock tracer(false);
        auto result = loader->LoadScene(tracer, "Scenes/Kitchen/KitchenGFG.json");
        EXPECT_FALSE(result.has_error());

        if(result.has_error())
        {
            MRAY_LOG("Err! :: {}", result.error().GetError());
            ASSERT_FALSE(true);
        }
        else all += result.value().loadTimeMS;

        loader->ClearScene();
    }

    MRAY_LOG("Average: {:.3f}ms", all / static_cast<double>(TOTAL_RUNS));
}