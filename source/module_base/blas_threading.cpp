#include "blas_threading.h"

#include <mutex>
#include <stdexcept>

#ifdef __MKL
#include <mkl_service.h>
#elif defined(__OPENBLAS)
extern "C"
{
int openblas_get_num_threads();
void openblas_set_num_threads(int num_threads);
}
#endif

namespace ModuleBase
{
namespace BlasThreading
{
namespace
{

std::recursive_mutex& control_mutex()
{
    static std::recursive_mutex mutex;
    return mutex;
}

} // namespace

bool is_control_available()
{
#if defined(__MKL) || defined(__OPENBLAS)
    return true;
#else
    return false;
#endif
}

int get_num_threads()
{
#ifdef __MKL
    return mkl_get_max_threads();
#elif defined(__OPENBLAS)
    return openblas_get_num_threads();
#else
    return 1;
#endif
}

void set_num_threads(const int num_threads)
{
    if (num_threads <= 0)
    {
        throw std::invalid_argument("BLAS thread count must be positive");
    }
#ifdef __MKL
    mkl_set_num_threads(num_threads);
#elif defined(__OPENBLAS)
    openblas_set_num_threads(num_threads);
#endif
}

class ScopedThreadLimit::Impl
{
  public:
    explicit Impl(const int num_threads)
        : lock_(control_mutex(), std::defer_lock), previous_threads_(1), changed_(false)
    {
        if (num_threads <= 0)
        {
            throw std::invalid_argument("BLAS thread count must be positive");
        }
        if (!is_control_available())
        {
            return;
        }
        lock_.lock();
        previous_threads_ = get_num_threads();
        changed_ = previous_threads_ != num_threads;
        if (changed_)
        {
            set_num_threads(num_threads);
        }
    }

    ~Impl()
    {
        if (changed_)
        {
            set_num_threads(previous_threads_);
        }
    }

  private:
    std::unique_lock<std::recursive_mutex> lock_;
    int previous_threads_;
    bool changed_;
};

ScopedThreadLimit::ScopedThreadLimit(const int num_threads) : impl_(new Impl(num_threads))
{
}

ScopedThreadLimit::~ScopedThreadLimit() = default;

} // namespace BlasThreading
} // namespace ModuleBase
