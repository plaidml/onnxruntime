// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#include "core/providers/plaidml/plaidml_execution_provider.h"
#include "test/providers/provider_test_utils.h"
#include "default_providers.h"
#include "gtest/gtest.h"
#include "core/session/inference_session.h"
#include "test/framework/test_utils.h"

namespace onnxruntime {
namespace test {

// TODO: This is based on the nGraph tests, investigate if we want something different
void RunTest(
    const std::string& model_path,
    const NameMLValMap& feeds,
    const std::vector<std::string>& output_names,
    const std::vector<std::vector<int64_t>>& expected_shapes,
    const std::vector<std::vector<float>>& expected_values,
    const Environment& env) {
  SessionOptions so;
  InferenceSession session_object(so, env);

  auto status = session_object.RegisterExecutionProvider(DefaultPlaidMLExecutionProvider());
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  status = session_object.Load(model_path);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  RunOptions run_options{};
  run_options.run_tag = "PlaidML EP test tag";
  run_options.run_log_verbosity_level = 1;

  std::vector<OrtValue> fetches;
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Run failed with status: " << status.ErrorMessage();
    return;
  }

  for (size_t idx = 0; idx < expected_values.size(); ++idx) {
    auto& got_tensor = fetches[idx].Get<Tensor>();
    auto* got = got_tensor.Data<float>();
    auto& expected = expected_values[idx];
    TensorShape expected_shape(expected_shapes[idx]);
    EXPECT_EQ(got_tensor.Shape(), expected_shape);
    for (size_t i = 0; i < expected.size(); i++) {
      EXPECT_EQ(got[i], expected[i]);
    }
  }
}

// TODO: This is separated out as in nGraph, but I think we may want to make it integrated
void add_feeds(NameMLValMap& feeds, std::string name, std::vector<int64_t> dims, std::vector<float> value) {
  OrtValue ml_value;

  auto allocator = TestPlaidMLExecutionProvider()->GetAllocator(0, OrtMemTypeDefault);
  ASSERT_TRUE(allocator) << "Attempting to get allocator for input tensors yielded null pointer";
  CreateMLValue<float>(allocator, dims, value, &ml_value);

  feeds.insert(std::make_pair(name, ml_value));
}

TEST(PlaidMLExecutionProviderTest, Basic_Test) {
  NameMLValMap feeds;
  add_feeds(feeds, "A", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
  add_feeds(feeds, "B", {4}, {2.0f, 2.0f, 2.0f, 2.0f});

  std::vector<std::vector<float>> expected_values = {
      {4.0f, 8.0f, 12.0f, 16.0f}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  // TODO: We're borrowing nGraph test data just to get something basic going, but switch to our own network & inputs eventually
  RunTest("testdata/ngraph/Basic_Test.onnx", feeds, {"Z"}, expected_shapes, expected_values, GetEnvironment());
}

TEST(PlaidMLExecutionProviderTest, AddTest)
{
  NameMLValMap feeds;
  add_feeds(feeds, "A", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
  add_feeds(feeds, "B", {4}, {2.0f, 2.0f, 2.0f, 2.0f});

  std::vector<std::vector<float>> expected_values = {
      {3.0f, 4.0f, 5.0f, 6.0f}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/plaidml/add.onnx", feeds, {"Z"}, expected_shapes, expected_values, GetEnvironment());
}

TEST(PlaidMLExecutionProviderTest, CeilTest)
{
  NameMLValMap feeds;

  add_feeds(feeds, "A", {4}, {-4.5f, -2.2f, 3.7f, 4.1f});

  std::vector<std::vector<float>> expected_values = {
      {-4.0f, -2.0f, 4.0f, 5.0f}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/plaidml/ceil.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
}

// TEST(PlaidMLExecutionProviderTest, ClipTest)
// {
//   NameMLValMap feeds;

//   add_feeds(feeds, "A", {4}, {-4.0f, -2.0f, 3.0f, 4.0f});
//   add_feeds(feeds, "min_val", {1}, {-1.0f});
//   add_feeds(feeds, "max_val", {1}, {1.0f});
//   std::vector<std::vector<float>> expected_values = {
//       {-1.0f, -1.0f, 1.0f, 1.0f}};

//   std::vector<std::vector<int64_t>> expected_shapes = {
//       {4}};

//   RunTest("testdata/plaidml/clip.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
// }

// TODO: fix -> in plaidml  "Invalid enumeration value"
//onnx nchw plaidml nhwc (default)
TEST(PlaidMLExecutionProviderTest, ConvTest)
{
  NameMLValMap feeds;

  add_feeds(feeds, "x", {1,1,5,5}, {0., 1., 2., 3., 4.,5.,
                         6., 7., 8., 9.,10., 
                         11., 12., 13., 14.,15., 
                         16., 17., 18., 19.,20., 
                         21., 22., 23., 24.});

    add_feeds(feeds, "w", {1,1,3,3}, {1., 1., 1.,
                                      1., 1., 1.,
                                      1., 1., 1.});


  std::vector<std::vector<float>> expected_values = {
      {12., 21., 27., 33., 24.,
       33., 54., 63., 72., 51.,
       63., 99., 108., 117., 81.,
       93., 144., 153., 162., 111.,
       72., 111., 117., 123., 84.}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {1,1,5,5}};

  RunTest("testdata/plaidml/conv.onnx", feeds, {"y"}, expected_shapes, expected_values, GetEnvironment());
}

TEST(PlaidMLExecutionProviderTest, CosTest)
{
  NameMLValMap feeds;

  add_feeds(feeds, "A", {4}, {-1.0f, 0.0f, 1.0f, 3.141592653589793f});

  std::vector<std::vector<float>> expected_values = {
      {0.5403023, 1.0, 0.5403023,-1.0}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/plaidml/cos.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
}

TEST(PlaidMLExecutionProviderTest, DivTest)
{
  NameMLValMap feeds;
  add_feeds(feeds, "A", {4}, {4.0f, 8.0f, 3.0f, 4.0f});
  add_feeds(feeds, "B", {4}, {2.0f, 2.0f, 3.0f, 2.0f});

  std::vector<std::vector<float>> expected_values = {
      {2.0f, 4.0f, 1.0f, 2.0f}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/plaidml/div.onnx", feeds, {"Z"}, expected_shapes, expected_values, GetEnvironment());
}

TEST(PlaidMLExecutionProviderTest, ExpTest)
{
  NameMLValMap feeds;

  add_feeds(feeds, "A", {4}, {-1.5255321 ,  0.49402064, 0.0, 2.3025851});

  std::vector<std::vector<float>> expected_values = {
      {0.21750529,  1.6388924 ,  1.0,  10.0}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/plaidml/exp.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
}

TEST(PlaidMLExecutionProviderTest, FlattenTest)
{
  NameMLValMap feeds;

  add_feeds(feeds, "A", {5,4,3}, {0.43511736, 0.684851  , 0.7479755,
                                  0.7615042 , 0.64659685, 0.2486658,
                                  0.65377426, 0.9870222 , 0.70429796,
                                  0.42194828, 0.84952426, 0.71247995,
                                  0.13712448, 0.50205755, 0.0216208,
                                  0.9232303 , 0.7301768 , 0.77392966,
                                  0.12261592, 0.6070028 , 0.41062224,
                                  0.56569165, 0.4989046 , 0.24323109,
                                  0.817748  , 0.28690833, 0.22093466,
                                  0.6299816 , 0.38470078, 0.63924766,
                                  0.3930874 , 0.82013094, 0.7363478,
                                  0.43261915, 0.35227805, 0.08362589,
                                  0.19125114, 0.47730285, 0.49478233,
                                  0.8514512 , 0.28996357, 0.62918735,
                                  0.29403505, 0.5220902 , 0.24513079,
                                  0.60193336, 0.9612385 , 0.36742613,
                                  0.98025334, 0.3123538 , 0.34941122,
                                  0.29762167, 0.24541792, 0.8721895,
                                  0.2395823 , 0.32739183, 0.37540108,
                                  0.42139563, 0.75938505, 0.48045367 });

  std::vector<std::vector<float>> expected_values = {
      {0.43511736, 0.684851  , 0.7479755 , 0.7615042 , 0.64659685,
        0.2486658 , 0.65377426, 0.9870222 , 0.70429796, 0.42194828,
        0.84952426, 0.71247995,
       0.13712448, 0.50205755, 0.0216208 , 0.9232303 , 0.7301768 ,
        0.77392966, 0.12261592, 0.6070028 , 0.41062224, 0.56569165,
        0.4989046 , 0.24323109,
       0.817748  , 0.28690833, 0.22093466, 0.6299816 , 0.38470078,
        0.63924766, 0.3930874 , 0.82013094, 0.7363478 , 0.43261915,
        0.35227805, 0.08362589,
       0.19125114, 0.47730285, 0.49478233, 0.8514512 , 0.28996357,
        0.62918735, 0.29403505, 0.5220902 , 0.24513079, 0.60193336,
        0.9612385 , 0.36742613,
       0.98025334, 0.3123538 , 0.34941122, 0.29762167, 0.24541792,
        0.8721895 , 0.2395823 , 0.32739183, 0.37540108, 0.42139563,
        0.75938505, 0.48045367}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {5,12}};

  RunTest("testdata/plaidml/flatten.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
}

TEST(PlaidMLExecutionProviderTest, FloorTest)
{
  NameMLValMap feeds;

  add_feeds(feeds, "A", {4}, {-1.5 ,  0.4, 0.5, 2.3});

  std::vector<std::vector<float>> expected_values = {
      {-2.,  0.,  0.,  2.}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/plaidml/floor.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
}

TEST(PlaidMLExecutionProviderTest, LogTest)
{
  NameMLValMap feeds;

  add_feeds(feeds, "A", {4}, {0.21750529,  1.6388924 ,  1.0,  10.0});

  std::vector<std::vector<float>> expected_values = {
      {-1.5255321 ,  0.49402064, 0.0, 2.3025851}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/plaidml/log.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
}

TEST(PlaidMLExecutionProviderTest, MulTest)
{
  NameMLValMap feeds;
  // TODO: fix dimension mismatch issue
  add_feeds(feeds, "A", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
  add_feeds(feeds, "B", {4}, {2.0f, 2.0f, 2.0f, 2.0f});

  std::vector<std::vector<float>> expected_values = {
      {2.0f, 4.0f, 6.0f, 8.0f}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/plaidml/mul.onnx", feeds, {"Z"}, expected_shapes, expected_values, GetEnvironment());
}

TEST(PlaidMLExecutionProviderTest, ReluTest)
{
  NameMLValMap feeds;
  //TODO: remove hardcoded values add expected value computation 
  add_feeds(feeds, "X", {3,4,5}, {0.9105323 , -0.17202467,  0.19540711,  0.5057067 , 0.05839625,
                                  0.4989552 , -0.24061275,  0.31148124, -0.25042275, -1.1431444, 
                                  0.6542162 , -1.7285362 ,  0.88606644,  1.6676011 ,-1.3186206, 
                                  -2.2778082 , -0.77277696, -0.8941763 , -0.71552706, -1.1515434,
                                  0.12072674, -1.3711382 ,  0.37264553,  1.2529551 ,-0.33971852,
                                  -0.7263648 ,  0.9757802 , -0.51076573, -1.2931641 ,0.06053324,
                                  -0.05999891,  0.6783777 , -0.88862735, -0.12444367, -0.71822774,
                                  -0.17955114, -0.18401518,  1.7149601 ,  0.99298984, -0.560877,
                                  -0.04585629, -0.48615912,  1.1301773 ,  0.93092895, 0.42474958,
                                  -1.560841  , -0.13032605, -0.12858576,  0.6732823 ,0.27759868,
                                  0.05039625,  0.1503488 , -0.07602423,  0.24816841, 0.7897034,
                                  -0.5707704 ,  2.068636  ,  1.2006909 , -0.7577646 ,-1.5209451});

  std::vector<std::vector<float>> expected_values = {
      {0.9105323 , 0.        , 0.19540711, 0.5057067 , 0.05839625,
      0.4989552 , 0.        , 0.31148124, 0.        , 0.,
      0.6542162 , 0.        , 0.88606644, 1.6676011 , 0.,
      0.        , 0.        , 0.        , 0.        , 0.,
      0.12072674, 0.        , 0.37264553, 1.2529551 , 0.,
      0.        , 0.9757802 , 0.        , 0.        , 0.06053324,
      0.        , 0.6783777 , 0.        , 0.        , 0.,
      0.        , 0.        , 1.7149601 , 0.99298984, 0.,
      0.        , 0.        , 1.1301773 , 0.93092895, 0.42474958,
      0.        , 0.        , 0.        , 0.6732823 , 0.27759868,
      0.05039625, 0.1503488 , 0.        , 0.24816841, 0.7897034 ,
      0.        , 2.068636  , 1.2006909 , 0.        , 0.}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {3,4,5}};

  RunTest("testdata/plaidml/relu.onnx", feeds, {"Y"}, expected_shapes, expected_values, GetEnvironment());
}

TEST(PlaidMLExecutionProviderTest, SigmoidTest)
{
  NameMLValMap feeds;
  //TODO: remove hardcoded values add expected value computation 
  add_feeds(feeds, "A", {4}, {1.0f, 2.0f, 3.0f, 4.0f});

  std::vector<std::vector<float>> expected_values = {
      {0.7310586 , 0.880797  , 0.95257413, 0.98201376 }};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/plaidml/sigmoid.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
}

TEST(PlaidMLExecutionProviderTest, SoftmaxTest)
{
  NameMLValMap feeds;

  add_feeds(feeds, "A", {1,3}, {-1.0f,  0.0f,  1.0f});

  std::vector<std::vector<float>> expected_values = {
      {0.09003057, 0.24472847, 0.66524096 }};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {1,3}};

   RunTest("testdata/plaidml/softmax.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
}

TEST(PlaidMLExecutionProviderTest, SqrtTest)
{
  NameMLValMap feeds;

  add_feeds(feeds, "A", {4}, {4.0f, 9.0f, 4.0f, 16.0f});

  std::vector<std::vector<float>> expected_values = {
      {2.0, 3.0, 2.0,4.0}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/plaidml/sqrt.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
}

TEST(PlaidMLExecutionProviderTest, SubTest)
{
  NameMLValMap feeds;
  add_feeds(feeds, "A", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
  add_feeds(feeds, "B", {4}, {2.0f, 2.0f, 2.0f, 2.0f});

  std::vector<std::vector<float>> expected_values = {
      {-1.0f, 0.0f, 1.0f, 2.0f}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/plaidml/sub.onnx", feeds, {"Z"}, expected_shapes, expected_values, GetEnvironment());
}


TEST(PlaidMLExecutionProviderTest, TransposeTest)
{
  NameMLValMap feeds;
  //TODO: remove hardcoded values add expected value computation 
  add_feeds(feeds, "A", {2,3,4}, {0.20217323, 0.7418437 , 0.8095215 , 0.78543824,
                                  0.8493133 , 0.54738104, 0.5216761 , 0.5134987 ,
                                  0.8872334 , 0.8586917 , 0.64419013, 0.17729734,
                                  0.81542754, 0.14297868, 0.42068878, 0.96306694,
                                  0.9126632 , 0.11507355, 0.8961067 , 0.90747726,
                                  0.54890287, 0.03705996, 0.56190175, 0.8113009 });

  std::vector<std::vector<float>> expected_values = {
      {0.20217323, 0.81542754,
        0.8493133 , 0.9126632,
        0.8872334 , 0.54890287,
        0.7418437 , 0.14297868,
        0.54738104, 0.11507355,
        0.8586917 , 0.03705996,
        0.8095215 , 0.42068878,
        0.5216761 , 0.8961067,
        0.64419013, 0.56190175,
        0.78543824, 0.96306694,
        0.5134987 , 0.90747726,
        0.17729734, 0.8113009}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4,3,2}};

  RunTest("testdata/plaidml/transpose.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
}
}  // namespace test
}  // namespace onnxruntime
