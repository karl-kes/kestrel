#include <kestrel/enums.hpp>
#include <kestrel/field.hpp>
#include <kestrel/grid.hpp>
#include <kestrel/precision.hpp>
#include <kestrel/timer.hpp>

auto main() -> int {
  constexpr kestrel::grid<3> geometry{
    {10, 10, 10},
    {1.0, 1.0, 1.0}
  };

  kestrel::scalar_field<kestrel::fp_t, 3uz> scalar{geometry.cells()};

  auto field{scalar.view()};
  field(1, 1, 1) = 0.0;

  return 0;
}
