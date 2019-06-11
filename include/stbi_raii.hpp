#if !defined(SPLIT_HOST_INCLUDED_STBI_RAII)
#define SPLIT_HOST_INCLUDED_STBI_RAII

#include <memory>
#include <gsl/gsl-lite.hpp>
#include <nonstd/value_ptr.hpp>


namespace stbi
{
using real = float;
/***
   @brief An RAII wrapper for an array of data which represents the pixels of an
   image. The owned memory is freed using stb_image_free.
   ***/
class ScopedImage
{
public:
  /***
     @brief Takes ownership of the supplied array of data.
     @param i_image_ptr A pointer to an array of image pixel data.
     @param i_width The width in pixels of the image.
     @param i_height The height in pixels of the image.
     @param i_nchannels The number of color channels the image contains.
   ***/
  ScopedImage(gsl::not_null<real*> i_image_ptr,
              int i_width,
              int i_height,
              int i_nchannels);
  ///@brief Default copy constructor performs a deep copy of the image data.
  ScopedImage(const ScopedImage&);
  ///@brief Default copy assignment operator performs a deep copy of the image
  /// data.
  ScopedImage& operator=(const ScopedImage&);
  ///@brief Default move constructor relinquishes ownership of the image data.
  ScopedImage(ScopedImage&&);
  ///@brief Default move assignment operator relinquishes ownership of the image
  /// data.
  ScopedImage& operator=(ScopedImage&&);
  ///@brief Destructor frees the image data using stbi_image_free.
  ~ScopedImage();

  ///@brief Obtains a mutable pointer to the image pixel data.
  real* get() noexcept;
  ///@brief Obtains an immutable pointer to the image pixel data.
  const real* get() const noexcept;
  ///@brief The width of the image in pixels.
  int width() const noexcept;
  ///@brief The height of the image in pixels.
  int height() const noexcept;
  ///@brief The number of color channels stored in the image.
  int n_channels() const noexcept;
  ///@brief The number of pixels in the image.
  int n_pixels() const noexcept;
  ///@brief The total number of color channels in the image (npixels*nchannels).
  int n_pixel_data() const noexcept;

private:
  ///@brief Forward declare implementation.
  struct ScopedImageImpl;
  ///@brief Pointer to implementation.
  nonstd::value_ptr<ScopedImageImpl> m_impl;
};

/***
   @brief Wraps a call to stbi_loadf, and returns the loaded image inside of an
   RAII class.
   @param i_path The file path to load our image from.
   @param i_desired_channels If this parameter is not zero, we override the
   number of channels stored in the image, and keep the first channels as
   specified by the parameter.
 ***/
ScopedImage loadf(gsl::czstring i_path, int i_desired_channels = 0);

/***
   @brief Wraps a call to stbi_write_png, which writes the provided image to
   disk at the specified file path.
   @param i_path The file path to write our image to.
   @param i_image The image to write to disk.
 ***/
void writef(gsl::czstring i_path, const ScopedImage& i_image); 
std::vector<uint8_t> quantize(const ScopedImage& i_image);
void gamma_correct(const ScopedImage& i_image, real* o_image, const real gamma);
}  // namespace stbi


#endif  // SPLIT_HOST_INCLUDED_STBI_RAII

