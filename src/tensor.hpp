#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <array>
#include <functional>

namespace ame {

// ND tensor:

template <typename m_type, std::size_t m_shape_head,
          std::size_t... m_shape_tail>
class tensor {
public:
  tensor() : m_content() {}
  tensor(
      std::array<tensor<m_type, m_shape_tail...>, m_shape_head> const &content)
      : m_content(content) {}

  template <std::size_t depth = 0> std::size_t size() {
    if constexpr (depth == 0)
      return m_shape_head;
    else
      return m_content[0].template size<depth - 1>();
  }

  auto const &operator[](std::size_t index) const { return m_content[index]; }
  auto &operator[](std::size_t index) { return m_content[index]; }
  operator std::array<tensor<m_type, m_shape_tail...>, m_shape_head>() const {
    return m_content;
  }

  template <std::size_t size> auto sub(std::size_t i1) {
    tensor<m_type, size, m_shape_tail...> result;
    for (std::size_t i = 0; i < size; ++i) {
      result[i] = {{(*this)[i1 + i]}};
    }
    return result;
  }

  tensor<m_type, m_shape_head, m_shape_tail...>
  operator+(tensor<m_type, m_shape_head, m_shape_tail...> const &other) const {
    tensor<m_type, m_shape_head, m_shape_tail...> result;
    for (std::size_t i = 0; i < m_shape_head; ++i)
      result[i] = (*this)[i] + other[i];
    return result;
  }
  tensor<m_type, m_shape_head, m_shape_tail...> &
  operator+=(tensor<m_type, m_shape_head, m_shape_tail...> const &other) {
    for (std::size_t i = 0; i < m_shape_head; ++i)
      (*this)[i] += other[i];
    return *this;
  }
  tensor<m_type, m_shape_head, m_shape_tail...>
  operator-(tensor<m_type, m_shape_head, m_shape_tail...> const &other) const {
    tensor<m_type, m_shape_head, m_shape_tail...> result;
    for (std::size_t i = 0; i < m_shape_head; ++i)
      result[i] = (*this)[i] - other[i];
    return result;
  }
  tensor<m_type, m_shape_head, m_shape_tail...> &
  operator-=(tensor<m_type, m_shape_head, m_shape_tail...> const &other) {
    for (std::size_t i = 0; i < m_shape_head; ++i)
      (*this)[i] -= other[i];
    return *this;
  }

  tensor<m_type, m_shape_head, m_shape_tail...>
  map(std::function<m_type(m_type)> f) const {
    tensor<m_type, m_shape_head, m_shape_tail...> result;
    for (std::size_t i = 0; i < m_shape_head; ++i)
      result[i] = m_content[i].map(f);
    return result;
  }

  m_type reduce_to_sum() {
    m_type result = m_type();
    for (std::size_t i = 0; i < m_shape_head; ++i)
      result += (*this)[i].reduce_to_sum();
    return result;
  }

protected:
  std::array<tensor<m_type, m_shape_tail...>, m_shape_head> m_content;
};

// 1D tensor:

template <typename m_type, std::size_t m_shape_head>
class tensor<m_type, m_shape_head> {
public:
  tensor() {}
  tensor(std::array<m_type, m_shape_head> const &content)
      : m_content(content) {}

  template <std::size_t depth = 0> std::size_t size() {
    static_assert(depth == 0, "Sorry, but the depth given is too high !");
    return m_shape_head;
  }

  auto const &operator[](std::size_t index) const { return m_content[index]; }
  auto &operator[](std::size_t index) { return m_content[index]; }
  operator std::array<m_type, m_shape_head>() const { return m_content; }

  template <std::size_t size> auto sub(std::size_t i1) {
    tensor<m_type, size> result;
    for (std::size_t i = 0; i < size; ++i) {
      result[i] = {{(*this)[i1 + i]}};
    }
    return result;
  }

  tensor<m_type, m_shape_head>
  operator+(tensor<m_type, m_shape_head> const &other) const {
    tensor<m_type, m_shape_head> result;
    for (std::size_t i = 0; i < m_shape_head; ++i)
      result[i] = (*this)[i] + other[i];
    return result;
  }
  tensor<m_type, m_shape_head> &
  operator+=(tensor<m_type, m_shape_head> const &other) {
    for (std::size_t i = 0; i < m_shape_head; ++i)
      (*this)[i] += other[i];
    return *this;
  }
  tensor<m_type, m_shape_head>
  operator-(tensor<m_type, m_shape_head> const &other) const {
    tensor<m_type, m_shape_head> result;
    for (std::size_t i = 0; i < m_shape_head; ++i)
      result[i] = (*this)[i] - other[i];
    return result;
  }
  tensor<m_type, m_shape_head> &
  operator-=(tensor<m_type, m_shape_head> const &other) {
    for (std::size_t i = 0; i < m_shape_head; ++i)
      (*this)[i] -= other[i];
    return *this;
  }

  tensor<m_type, m_shape_head> map(std::function<m_type(m_type)> f) const {
    tensor<m_type, m_shape_head> result;
    for (std::size_t i = 0; i < m_shape_head; ++i)
      result[i] = f(m_content[i]);
    return result;
  }

  m_type reduce_to_sum() {
    m_type result = m_type();
    for (std::size_t i = 0; i < m_shape_head; ++i)
      result += (*this)[i];
    return result;
  }

protected:
  std::array<m_type, m_shape_head> m_content;
};

template <typename T, std::size_t h, std::size_t w>
using matrix = tensor<T, h, w>;
template <typename T, std::size_t s> using vector_row = tensor<T, 1, s>;
template <typename T, std::size_t s> using vector_col = tensor<T, s, 1>;

template <typename T, std::size_t n, std::size_t m, std::size_t p>
matrix<T, n, p> operator*(matrix<T, n, m> const &m1,
                          matrix<T, m, p> const &m2) {
  matrix<T, n, p> result;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < p; ++j)
      for (std::size_t k = 0; k < m; ++k)
        result[i][j] += m1[i][k] * m2[k][j];
  return result;
}
template <typename T, std::size_t h, std::size_t w>
matrix<T, w, h> operator~(matrix<T, h, w> const &m) {
  matrix<T, w, h> result;
  for (std::size_t i = 0; i < h; ++i)
    for (std::size_t j = 0; j < w; ++j)
      result[j][i] = m[i][j];
  return result;
}

} // namespace ame

#endif // TENSOR_HPP
