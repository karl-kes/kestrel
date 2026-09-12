#pragma once

#include <array>
#include <limits>
#include <stdexcept>

#include <xpu/soa.hpp>

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

} // namespace kestrel
