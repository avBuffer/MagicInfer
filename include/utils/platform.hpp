#ifndef MAGIC_LAYER_DETAILS_PLATFORM_HPP_
#define MAGIC_LAYER_DETAILS_PLATFORM_HPP_

#define MAGIC_FORCE_INLINE 1

#if MAGIC_FORCE_INLINE

#ifdef _MSC_VER
#define MAGIC_FORCEINLINE __forceinline

#elif defined(__GNUC__)
#define MAGIC_FORCEINLINE inline __attribute__((__always_inline__))

#elif defined(__CLANG__)

#if __has_attribute(__always_inline__)
#define MAGIC_FORCEINLINE inline __attribute__((__always_inline__))
#else
#define MAGIC_FORCEINLINE inline
#endif

#else
#define MAGIC_FORCEINLINE inline
#endif

#else
#define MAGIC_FORCEINLINE inline
#endif

#endif //MAGIC_LAYER_DETAILS_PLATFORM_HPP_
