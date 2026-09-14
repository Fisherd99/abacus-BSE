#ifndef MODULE_BASE_BLAS_THREADING_H_
#define MODULE_BASE_BLAS_THREADING_H_

#include <memory>

namespace ModuleBase
{
namespace BlasThreading
{

/// Return whether the linked BLAS exposes a supported thread-control API.
bool is_control_available();

/// Return the current BLAS thread count, or 1 when no control API is available.
int get_num_threads();

/// Set the BLAS thread count when a supported control API is available.
void set_num_threads(int num_threads);

/// Temporarily set the BLAS thread count and restore it on scope exit.
///
/// MKL and OpenBLAS change process-wide state through the APIs used here. The
/// implementation therefore serializes overlapping guards, and supports
/// nested guards from the same thread.
class ScopedThreadLimit
{
  public:
    explicit ScopedThreadLimit(int num_threads);
    ~ScopedThreadLimit();

    ScopedThreadLimit(const ScopedThreadLimit&) = delete;
    ScopedThreadLimit& operator=(const ScopedThreadLimit&) = delete;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace BlasThreading
} // namespace ModuleBase

#endif // MODULE_BASE_BLAS_THREADING_H_
