#include "../common/fit.h"

#include <cstddef>
#include <cstdint>

int main() {
    constexpr int64_t MiB = 1024 * 1024;
    constexpr size_t model = 9'685 * MiB;
    constexpr int64_t recurrent_bytes = 1'496 * MiB;
    constexpr size_t recurrent = static_cast<size_t>(recurrent_bytes);
    constexpr size_t compute = 810 * MiB;
    constexpr int64_t free_before_context = 15'847 * MiB;

    // Synthetic device data matching the ksia fit: the recurrent buffer is
    // part of the projection, but is not yet part of the device free value.
    const int64_t projected_free_before = common_fit_projected_free(
        free_before_context, model, recurrent, compute);
    const int64_t projected_free_after = common_fit_projected_free(
        free_before_context - recurrent, model, recurrent, compute);

    return projected_free_before == 3'856 * MiB &&
                   projected_free_after == projected_free_before - recurrent_bytes
               ? 0
               : 1;
}
