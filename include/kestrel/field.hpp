#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>

#include <xpu/config.hpp>
#include <xpu/launch.hpp>
#include <xpu/soa.hpp>

#include <kestrel/concepts.hpp>

namespace kestrel {

template <std::size_t Dimensions>
class layout {
  static_assert(Dimensions > 0uz);

public:
  using extent = std::array<std::size_t, Dimensions>;
  using index_t = xpu::array<std::size_t, Dimensions>;

private:
  index_t cells_;
  index_t strides_;
  std::size_t count_;

public:
  explicit constexpr layout(extent cells)
    : cells_{}
    , strides_{}
    , count_{1uz}
  {
    for (auto d{0uz}; d < Dimensions; ++d) {
      const auto length{cells[d]};

      const auto empty_extent{length == 0uz};
      if (empty_extent) {
        throw std::invalid_argument{
          "Layout extents must be positive."
        };
      }

      const auto overflow{length > std::numeric_limits<std::size_t>::max() / count_};
      if (overflow) {
        throw std::overflow_error{
          "Layout size overflow."
        };
      }

      cells_[d] = length;
      strides_[d] = count_;
      count_ *= length;
    }
  }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto cells() const noexcept -> index_t {
    return cells_;
  }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto strides() const noexcept -> index_t {
    return strides_;
  }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto count() const noexcept -> std::size_t {
    return count_;
  }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto offset(const index_t& idx) const noexcept -> std::size_t {
    auto offset{0uz};

    for (auto d{0uz}; d < Dimensions; ++d) {
      assert(idx[d] < cells_[d]);
      offset += idx[d] * strides_[d];
    }

    return offset;
  }
};

template <arithmetic T, std::size_t Dimensions, std::size_t Components>
struct field_view {
  static_assert(Components > 0, "A field must have at least one component.");

  layout<Dimensions> mapping;
  xpu::soa_view<T, Components> data;

  using index_t = typename layout<Dimensions>::index_t;
};

template <arithmetic T, std::size_t Dimensions, std::size_t Components>
class field {
  static_assert(Components > 0);

public:
  using extent = typename layout<Dimensions>::extent;

private:
  layout<Dimensions> layout_;
  xpu::soa<T, Components> storage_;

public:
  explicit field(extent cells)
    : layout_{cells}
    , storage_{layout_.count()}
  { }

  using const_view_t = field_view<const T, Dimensions, Components>;
  using view_t = field_view<T, Dimensions, Components>;

  [[nodiscard]]
  auto view() const noexcept -> const_view_t {
    return {layout_, storage_.view()};
  }

  [[nodiscard]]
  auto view() noexcept -> view_t {
    return {layout_, storage_.view()};
  }
};

template <arithmetic T, std::size_t Dimensions>
using scalar_field = field<T, Dimensions, 1uz>;

template <arithmetic T, std::size_t Dimensions>
using vector_field = field<T, Dimensions, 3uz>;

} // namespace kestrel
