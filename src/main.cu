#include <xpu/launch.hpp>

#include <kestrel/enums.hpp>
#include <kestrel/field.hpp>
#include <kestrel/grid.hpp>
#include <kestrel/precision.hpp>
#include <kestrel/timer.hpp>

template <std::size_t Dimensions>
using domain_t = xpu::range<Dimensions>;

auto main() -> int {
  constexpr kestrel::grid<3> geometry{
    {100, 100, 100},
    {1.0, 1.0, 1.0}
  };

  domain_t<3> domain_3d{
    {0, 0, 0},
    {10, 10, 10},
    {1, 1, 1}
  };

  using scalar_field = kestrel::scalar_field<kestrel::fp_t, 3uz>;
  using vector_field = kestrel::vector_field<kestrel::fp_t, 3uz>;

  scalar_field scalar{geometry.cells()};

  auto field{scalar.view()};
  field(1, 1, 1) = 0.0;

  xpu::parallel_for(domain_3d, [field](const auto& idx) mutable {
    field(idx) = 0.0;
  });

  vector_field vec{geometry.cells()};

  auto vec_field{vec.view()};
  vec_field[kestrel::axis::x](1, 1, 1) = 0.0;

  xpu::parallel_for(domain_3d, [vec_field](const auto& idx) mutable {
    vec_field[kestrel::axis::x](idx) = 0.0;
    vec_field[kestrel::axis::y](idx) = 0.0;
    vec_field[kestrel::axis::z](idx) = 0.0;
  });

  return 0;
}
