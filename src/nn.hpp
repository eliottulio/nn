#ifndef NN_HPP
#define NN_HPP

#include "layer.hpp"
#include "tensor.hpp"

namespace ame {

// N-layer NN:

template <std::size_t m_input_nb, std::size_t m_first_hidden_input_size,
          std::size_t... m_other_hidden_sizes>
class nn {
public:
  constexpr std::size_t input_nb() { return m_input_nb; }
  static constexpr std::size_t output_nb() {
    return nn<m_first_hidden_input_size, m_other_hidden_sizes...>::output_nb();
  }

  template <std::size_t data_size = 1>
  auto feed_forward(matrix<double, data_size, m_input_nb> inputs) {
    return m_other_layers.feed_forward(m_first_layer.feed_forward(inputs));
  }
  template <std::size_t data_size = 1>
  auto train(matrix<double, data_size, m_input_nb> inputs,
             matrix<double, data_size, output_nb()> expected_outputs,
             double learning_rate) {
    return m_first_layer.train(
        inputs,
        m_other_layers.train(m_first_layer.feed_forward(inputs),
                             expected_outputs, learning_rate),
        learning_rate);
  }

  template <std::size_t data_size = 1>
  double calc_error(matrix<double, data_size, m_input_nb> inputs,
                    matrix<double, data_size, output_nb()> outputs) {
    return (feed_forward(inputs) - outputs)
        .map([](double x) { return x * x; })
        .reduce_to_sum();
  }

protected:
  layer<m_input_nb, m_first_hidden_input_size> m_first_layer;
  nn<m_first_hidden_input_size, m_other_hidden_sizes...> m_other_layers;
};

// 1-layer NN:

template <std::size_t m_input_nb, std::size_t m_output_nb>
class nn<m_input_nb, m_output_nb> {
public:
  constexpr std::size_t input_nb() { return m_input_nb; }
  static constexpr std::size_t output_nb() { return m_output_nb; }

  template <std::size_t data_size = 1>
  auto feed_forward(matrix<double, data_size, m_input_nb> inputs) {
    return m_layer.feed_forward(inputs);
  }
  template <std::size_t data_size = 1>
  auto train(matrix<double, data_size, m_input_nb> inputs,
             matrix<double, data_size, m_output_nb> expected_outputs,
             double learning_rate) {
    return m_layer.train(inputs, expected_outputs, learning_rate);
  }

  template <std::size_t data_size = 1>
  double calc_error(matrix<double, data_size, m_input_nb> inputs,
                    matrix<double, data_size, m_output_nb> outputs) {
    return (feed_forward(inputs) - outputs)
        .map([](double x) { return x * x; })
        .reduce_to_sum();
  }

protected:
  layer<m_input_nb, m_output_nb> m_layer;
};

} // namespace ame

#endif // NN_HPP
