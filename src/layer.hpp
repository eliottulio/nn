#ifndef LAYER_HPP
#define LAYER_HPP

#include "tensor.hpp"
#include <cmath>

namespace {

template <std::size_t m_input_nb, std::size_t m_output_nb,
          std::size_t nb_of_examples = 1>
class gradient_storage {
public:
  ame::matrix<double, m_input_nb, m_output_nb> weights_gradient;
  ame::vector_row<double, m_output_nb> biases_gradient;
  ame::matrix<double, nb_of_examples, m_input_nb> inputs_gradient;
};

double activation(double n) {
  // if (n > 0)
  return n;
  // else
  //   return n * 0.1;
}
double inverse_of_activation(double n) {
  // if (n > 0)
  return n;
  // else
  //   return n / 0.1;
}

} // namespace

namespace ame {

template <std::size_t m_input_nb, std::size_t m_output_nb> class layer {
public:
  layer() {
    for (std::size_t i = 0; i < m_weights.template size<0>(); i++)
      for (std::size_t j = 0; j < m_weights.template size<1>(); j++)
        m_weights[i][j] = (rand() % 1000) * 2 / 999.0 - 1;
    for (std::size_t i = 0; i < m_biases.template size<1>(); i++)
      m_biases[0][i] = (rand() % 1000) * 2 / 999.0 - 1;
  }

  template <std::size_t data_size = 1>
  matrix<double, data_size, m_output_nb>
  feed_forward(matrix<double, data_size, m_input_nb> inputs) const {
    matrix<double, data_size, m_output_nb> result;
    for (std::size_t i = 0; i < data_size; i++)
      result[i] =
          ((ame::vector_row<double, m_input_nb>({{inputs[i]}}) * m_weights +
            m_biases))[0]
              .map(activation);
    return result;
  }

  template <std::size_t data_size = 1>
  double calc_error(matrix<double, data_size, m_input_nb> inputs,
                    matrix<double, data_size, m_output_nb> outputs) {
    return (feed_forward(inputs) - outputs)
        .map([](double x) { return x * x; })
        .reduce_to_sum();
  }

  gradient_storage<m_input_nb, m_output_nb>
  calculate_gradient(vector_row<double, m_input_nb> const &inputs,
                     vector_row<double, m_output_nb> expected_outputs) {

    expected_outputs = expected_outputs.map(inverse_of_activation);

    matrix<double, m_input_nb, m_output_nb> weights_gradient;
    vector_row<double, m_output_nb> biases_gradient;
    vector_row<double, m_input_nb> inputs_gradient;

    for (std::size_t a = 0; a < m_output_nb; a++) {
      biases_gradient[0][a] += m_biases[0][a];
      biases_gradient[0][a] -= expected_outputs[0][a];
      for (std::size_t i = 0; i < m_input_nb; i++)
        biases_gradient[0][a] += inputs[0][i] * m_weights[i][a];

      for (std::size_t i = 0; i < m_input_nb; i++) {
        weights_gradient[i][a] += m_biases[0][a];
        weights_gradient[i][a] -= expected_outputs[0][a];
        for (std::size_t j = 0; j < m_input_nb; j++)
          weights_gradient[i][a] += inputs[0][j] * m_weights[j][a];
        weights_gradient[i][a] *= inputs[0][i];
      }
    }

    for (std::size_t b = 0; b < m_input_nb; b++) {
      for (std::size_t a = 0; a < m_output_nb; a++) {
        double sub_calc = 0;

        sub_calc += m_biases[0][a];
        sub_calc -= expected_outputs[0][a];
        for (std::size_t i = 0; i < m_input_nb; i++)
          sub_calc += inputs[0][i] * m_weights[i][a];
        sub_calc *= m_weights[b][a];

        inputs_gradient[0][b] += sub_calc;
      }
    }

    return {weights_gradient, biases_gradient, inputs_gradient};
  }

  template <std::size_t data_size>
  gradient_storage<m_input_nb, m_output_nb, data_size>
  calculate_gradient_average(
      matrix<double, data_size, m_input_nb> inputs,
      matrix<double, data_size, m_output_nb> expected_outputs) {
    gradient_storage<m_input_nb, m_output_nb, data_size> result;
    for (std::size_t i = 0; i < data_size; i++) {
      auto result_part = calculate_gradient(
          vector_row<double, m_input_nb>({{inputs[i]}}),
          vector_row<double, m_output_nb>({{expected_outputs[i]}}));
      result.biases_gradient += result_part.biases_gradient;
      result.weights_gradient += result_part.weights_gradient;
      result.inputs_gradient[i] = result_part.inputs_gradient[0];
    }

    auto average_out = [](double x) { return x / double(data_size); };

    result.biases_gradient.map(average_out);
    result.weights_gradient.map(average_out);
    return result;
  }

  template <std::size_t data_size>
  matrix<double, data_size, m_input_nb>
  train(matrix<double, data_size, m_input_nb> inputs,
        matrix<double, data_size, m_output_nb> expected_outputs,
        double learning_rate) {
    std::function<double(double)> get_actual_gradient =
        [learning_rate](double x) { return x * -learning_rate; };

    auto gradients = calculate_gradient_average(inputs, expected_outputs);
    m_weights += gradients.weights_gradient.map(get_actual_gradient);
    m_biases += gradients.biases_gradient.map(get_actual_gradient);
    return (inputs.map(inverse_of_activation) +
            gradients.inputs_gradient.map(get_actual_gradient))
        .map(activation);
  }

  constexpr std::size_t input_nb() const { return m_input_nb; }
  constexpr std::size_t output_nb() const { return m_output_nb; }

protected:
  matrix<double, m_input_nb, m_output_nb> m_weights;
  vector_row<double, m_output_nb> m_biases;
};

} // namespace ame

#endif // LAYER_HPP
