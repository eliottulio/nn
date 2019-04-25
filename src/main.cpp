#include "layer.hpp"
#include "nn.hpp"
#include "tensor.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>

int main() {
  srand(unsigned(time(NULL)));

  ame::matrix<double, 10, 2> inputs(
      {ame::tensor<double, 2>({0, 0}), ame::tensor<double, 2>({0, 1}),
       ame::tensor<double, 2>({0, 2}), ame::tensor<double, 2>({0, 3}),
       ame::tensor<double, 2>({0, 4}), ame::tensor<double, 2>({1, 0}),
       ame::tensor<double, 2>({1, 1}), ame::tensor<double, 2>({1, 2}),
       ame::tensor<double, 2>({1, 3}), ame::tensor<double, 2>({1, 4})});
  ame::matrix<double, 10, 1> outputs(
      {ame::tensor<double, 1>({0}), ame::tensor<double, 1>({0.5}),
       ame::tensor<double, 1>({1}), ame::tensor<double, 1>({1.5}),
       ame::tensor<double, 1>({2}), ame::tensor<double, 1>({0.5}),
       ame::tensor<double, 1>({1}), ame::tensor<double, 1>({1.5}),
       ame::tensor<double, 1>({2}), ame::tensor<double, 1>({2.5})});

  ame::nn<2, 1> nn;

  constexpr std::size_t number_of_trainings = 1000;
  constexpr std::size_t batch_size = 10;
  // Must be an integer divider of training data size.

  for (std::size_t i = 0; i < number_of_trainings; i++) {
    std::size_t batch_id = i;
    nn.template train<batch_size>(
        inputs.sub<batch_size>(batch_id * batch_size % inputs.size()),
        outputs.sub<batch_size>(batch_id * batch_size % outputs.size()), 0.005);
    auto result = nn.feed_forward(
        inputs.sub<batch_size>(batch_id * batch_size % inputs.size()));
    std::cout << double(i + 1) * 100.0 / number_of_trainings
              << "% error: " << nn.calc_error(inputs, outputs) << '\n';
  }

  return 0;
}
