#include <sstream>

#include <gtest/gtest.h>
#include <BS/BS_thread_pool.hpp>

#include "SceneLoaderMRay/EntryPoint.h"

#include "Core/Types.h"
#include "Core/SharedLibrary.h"
#include "Core/SceneLoaderI.h"

#include "MockTracer.h"
#include "TestScenes.h"

class SceneLoaderMRayTest : public ::testing::Test
{
    protected:
    // Dunno where or when the destructor is called
    // So everything is wrapped on unique_ptrs
    std::unique_ptr<BS::thread_pool>    pool = nullptr;
    std::unique_ptr<SharedLibrary>      dllFile = nullptr;
    SharedLibPtr<SceneLoaderI>          loader = {nullptr, nullptr};

    void SetUp() override;
    void TearDown() override;
};

void SceneLoaderMRayTest::SetUp()
{
    pool = std::make_unique<BS::thread_pool>(1, [](){});
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
    EXPECT_TRUE(static_cast<bool>(result.first));
}

TEST_F(SceneLoaderMRayTest, MinimalValid)
{
    TracerMock tracer;
    std::istringstream ss{std::string(MinimalValidScene)};
    auto result = loader->LoadScene(tracer, ss);
    EXPECT_FALSE(static_cast<bool>(result.first));

    // Might as well print the error here
    if(result.first)
        MRAY_ERROR_LOG("{}", result.first.GetError());
}

TEST_F(SceneLoaderMRayTest, Basic)
{
    TracerMock tracer;
    std::istringstream ss{std::string(BasicScene)};
    auto result = loader->LoadScene(tracer, ss);
    EXPECT_FALSE(static_cast<bool>(result.first));

    if(result.first)
        MRAY_ERROR_LOG("{}", result.first.GetError());
}

TEST_F(SceneLoaderMRayTest, Kitchen)
{
    //while(true)
    for(uint32_t i = 0; i < 512; i++)
    //for(uint32_t i = 0; i < 1; i++)
    {
        TracerMock tracer(true);

        if(i > 0) SetUp();

        auto result = loader->LoadScene(tracer, "Kitchen/Kitchen.json");
        EXPECT_FALSE(static_cast<bool>(result.first));

        MRAY_LOG("{} --- {}ms",
                 result.first.GetError(),
                 result.second);

        TearDown();
    }
    //EXPECT_FALSE(result.first);

}