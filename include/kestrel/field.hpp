#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>

#include <xpu/config.hpp>
#include <xpu/launch.hpp>
#include <xpu/soa.hpp>

#include <kestrel/concepts.hpp>
#include <kestrel/enums.hpp>
#include <kestrel/layout.hpp>

namespace kestrel {

template <arithmetic T, std::size_t Dimensions>
struct component_view {
  using index_t = typename layout<Dimensions>::index_t;

  T* data;
  layout<Dimensions> mapping;

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto operator()(const index_t& idx) const noexcept -> T& {
    return data[mapping.offset(idx)];
  }

  template <std::integral... Indices>
    requires (coordinate_indices<Dimensions, Indices...>)
  [[nodiscard]] CUDA_CALLABLE
  constexpr auto operator()(Indices... indices) const noexcept -> T& {
    return (*this)(index_t{static_cast<std::size_t>(indices)...});
  }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto operator[](std::size_t idx) const noexcept -> T& {
    assert(idx < mapping.count());
    return data[idx];
  }
};

template <arithmetic T, std::size_t Dimensions, std::size_t Components>
struct field_view {
  static_assert(Components > 0, "A field must have at least one component.");

  using index_t = typename layout<Dimensions>::index_t;

  layout<Dimensions> mapping;
  xpu::soa_view<T, Components> data;

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto operator[](axis component) noexcept -> component_view<T, Dimensions> {
    const auto idx{to_index(component)};
    assert(idx < Components);

    return {data[idx], mapping};
  }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto operator[](axis component) const noexcept -> component_view<const T, Dimensions> {
    const auto idx{to_index(component)};
    assert(idx < Components);

    return {data[idx], mapping};
  }

  template <std::integral... Indices>
    requires(
      Components == 1uz &&
      coordinate_indices<Dimensions, Indices...>
    )
  [[nodiscard]] CUDA_CALLABLE
  constexpr auto operator()(Indices... indices) noexcept -> T& {
    return (*this)[axis::x](indices...);
  }

  template <std::integral... Indices>
    requires(
      Components == 1uz &&
      coordinate_indices<Dimensions, Indices...>
    )
  [[nodiscard]] CUDA_CALLABLE
  constexpr auto operator()(Indices... indices) const noexcept -> const T& {
    return (*this)[axis::x](indices...);
  }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto operator()(const index_t& idx) noexcept -> T&
    requires (Components == 1uz)
  {
    return (*this)[axis::x](idx);
  }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto operator()(const index_t& idx) const noexcept -> const T&
    requires (Components == 1uz)
  {
    return (*this)[axis::x](idx);
  }
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
