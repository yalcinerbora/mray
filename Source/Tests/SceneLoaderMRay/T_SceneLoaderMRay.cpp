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
    pool = std::make_unique<BS::thread_pool>(2, [](){});
    dllFile = std::make_unique<SharedLibrary>("SceneLoaderMRay");

    SharedLibArgs args
    {
        .mangledConstructorName = "ConstructSceneLoaderMRay",
        .mangledDestructorName = "DestroySceneLoaderMRay"
    };
    MRayError e = dllFile->GenerateObjectWithArgs<SceneLoaderConstructorArgs>(loader, args,
                                                                                     *pool);
    EXPECT_TRUE(!e);
}

void SceneLoaderMRayTest::TearDown()
{
    pool = nullptr;
    loader = {nullptr, nullptr};
    dllFile = nullptr;
}

TEST_F(SceneLoaderMRayTest, Basic)
{
    TracerMock tracer;

    std::istringstream ss{std::string(BasicScene)};
    auto result = loader->LoadScene(tracer, ss);

    MRAY_ERROR_LOG("{}", result.first.GetError());

    EXPECT_FALSE(result.first);
}

TEST_F(SceneLoaderMRayTest, Kitchen)
{
    //TracerMock tracer;
    //auto result = loader->LoadScene("Kitchen/Kitchen.json");
}