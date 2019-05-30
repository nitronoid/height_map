#include "stbi_raii.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


namespace
{
template <typename T>
T clamp(T v, T lo, T hi)
{
  return std::max(std::min(v, hi), lo);
}
/***
   @brief A deleter that frees memory allocated through the stb_image interface.
   ***/
template <typename T>
struct image_deleter
{
  void operator()(T* const ptr) const
  {
    if (ptr)
      stbi_image_free(ptr);
  }
};
}  // namespace

namespace stbi
{
struct ScopedImage::ScopedImageImpl
{
  ScopedImageImpl(real* i_image_ptr,
                  int i_width,
                  int i_height,
                  int i_nchannels)
    : m_image_ptr(i_image_ptr)
    , m_width(i_width)
    , m_height(i_height)
    , m_nchannels(i_nchannels)
  {
  }
  nonstd::value_ptr<real,
                    nonstd::vptr::detail::default_clone<real>,
                    image_deleter<real>>
    m_image_ptr = nullptr;
  int m_width;
  int m_height;
  int m_nchannels;
};

ScopedImage::ScopedImage(gsl::not_null<real*> i_image_ptr,
                         int i_width,
                         int i_height,
                         int i_nchannels)
  : m_impl(nonstd::make_value<ScopedImageImpl>(
      i_image_ptr, i_width, i_height, i_nchannels))
{
}

ScopedImage::ScopedImage(const ScopedImage&) = default;
ScopedImage& ScopedImage::operator=(const ScopedImage&) = default;
ScopedImage::ScopedImage(ScopedImage&&) = default;
ScopedImage& ScopedImage::operator=(ScopedImage&&) = default;
ScopedImage::~ScopedImage() = default;

real* ScopedImage::get() noexcept
{
  return m_impl->m_image_ptr.get();
}

const real* ScopedImage::get() const noexcept
{
  return m_impl->m_image_ptr.get();
}
int ScopedImage::width() const noexcept
{
  return m_impl->m_width;
}

int ScopedImage::height() const noexcept
{
  return m_impl->m_height;
}

int ScopedImage::n_channels() const noexcept
{
  return m_impl->m_nchannels;
}

int ScopedImage::n_pixels() const noexcept
{
  return m_impl->m_height * m_impl->m_width;
}
int ScopedImage::n_pixel_data() const noexcept
{
  return m_impl->m_height * m_impl->m_width * m_impl->m_nchannels;
}

ScopedImage loadf(gsl::czstring i_path, int i_desired_channels)
{
  int width, height, nchannels;
  auto raw_image =
    stbi_loadf(i_path, &width, &height, &nchannels, i_desired_channels);
  return ScopedImage(raw_image,
                     width,
                     height,
                     i_desired_channels ? i_desired_channels : nchannels);
}

void writef(gsl::czstring i_path, const ScopedImage& i_image)
{
  std::vector<uint8_t> quantized(i_image.n_pixel_data());
  std::transform(i_image.get(),
                 i_image.get() + i_image.n_pixel_data(),
                 quantized.begin(),
                 [](real v) {
                   const float gamma = 1.0f / 2.2f;
                   return pow(clamp(v, 0.f, 1.f), gamma) * 255.0f + 0.5f;
                 });

  stbi_write_png(i_path,
                 i_image.width(),
                 i_image.height(),
                 i_image.n_channels(),
                 quantized.data(),
                 i_image.n_channels() * i_image.width());
}
}  // namespace stbi

