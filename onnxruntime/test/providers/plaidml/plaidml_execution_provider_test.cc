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
    const std::vector<std::vector<float>>& expected_values) {
  SessionOptions so;
  InferenceSession session_object(so, &DefaultLoggingManager());

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
  RunTest("testdata/ngraph/Basic_Test.onnx", feeds, {"Z"}, expected_shapes, expected_values);
}

}  // namespace test
}  // namespace onnxruntime
