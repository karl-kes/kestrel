#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace kestrel {

template <typename T>
concept arithmetic = 
  std::floating_point<T> ||
  std::integral<T>;

template <typename T>
concept enumeration = 
  std::is_enum_v<T>;

template <std::size_t Dimensions, typename... Indices>
concept coordinate_indices =
  sizeof...(Indices) == Dimensions &&
  (std::integral<Indices> && ...);

} // namespace kestrel
