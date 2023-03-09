#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#define XSIMD_TEMPLATE template <typename T, typename A = default_arch>

// #define XSIMD_WITH_NEON 1

namespace xsimd {

struct generic16 {
  constexpr generic16() = default;
  static constexpr size_t alignment() noexcept {
    return 16;
  }
  static constexpr size_t size() noexcept {
    return 16;
  }
  static constexpr auto name() {
    return "compiler_autovec16";
  }
};

struct generic32 {
  constexpr generic32(const generic16&){};
  constexpr generic32() = default;
  static constexpr size_t alignment() noexcept {
    return 32;
  }
  static constexpr size_t size() noexcept {
    return 32;
  }
  static constexpr auto name() {
    return "compiler_autovec32";
  }
};

struct generic64 {
  constexpr generic64() = default;
  static constexpr size_t alignment() noexcept {
    return 64;
  }
  static constexpr size_t size() noexcept {
    return 64;
  }
  static constexpr auto name() {
    return "compiler_autovec64";
  }
};

#define USING_32_BYTE_VECTOR

struct sse2 : public generic16 {};
struct avx : public generic32 {};
struct avx2 : public generic32 {
  constexpr avx2() = default;
  constexpr avx2(const generic16&){};
};
struct avx512 : public generic64 {};
struct neon : public generic16 {};

struct half_vec {
  constexpr half_vec() = default;
  static constexpr size_t alignment() noexcept {
    return 8;
  }
  static constexpr size_t size() noexcept {
    return 8;
  }
  static constexpr auto name() {
    return "half_compiler_autovec";
  }
};

#ifdef __aarch64__
using generic = neon;
using default_arch = neon;
#endif

#ifdef __AVX2__
using generic = avx2;
using default_arch = avx2;
#endif

///////////////////////////////
///          TYPES          ///
///////////////////////////////

namespace types {
template <class T, class A>
struct has_simd_register : std::false_type {};

template <class T, class Arch>
struct simd_register {
  using vector_type = T;
  struct register_type {};
  register_type data;
};

#define XSIMD_DECLARE_SIMD_REGISTER(SCALAR_TYPE, ISA)               \
  template <>                                                       \
  struct simd_register<SCALAR_TYPE, ISA> {                          \
    using vector_type =                                             \
        std::array<SCALAR_TYPE, ISA::size() / sizeof(SCALAR_TYPE)>; \
    using register_type = vector_type;                              \
    register_type data;                                             \
    operator register_type() const noexcept {                       \
      return data;                                                  \
    }                                                               \
  };                                                                \
  template <>                                                       \
  struct has_simd_register<SCALAR_TYPE, ISA> : std::true_type {}

#define XSIMD_DECLARE_SIMD_REGISTERS(SCALAR_TYPE)      \
  XSIMD_DECLARE_SIMD_REGISTER(SCALAR_TYPE, generic16); \
  XSIMD_DECLARE_SIMD_REGISTER(SCALAR_TYPE, generic32); \
  XSIMD_DECLARE_SIMD_REGISTER(SCALAR_TYPE, generic64); \
  XSIMD_DECLARE_SIMD_REGISTER(SCALAR_TYPE, sse2);      \
  XSIMD_DECLARE_SIMD_REGISTER(SCALAR_TYPE, avx);       \
  XSIMD_DECLARE_SIMD_REGISTER(SCALAR_TYPE, avx2);      \
  XSIMD_DECLARE_SIMD_REGISTER(SCALAR_TYPE, avx512);    \
  XSIMD_DECLARE_SIMD_REGISTER(SCALAR_TYPE, neon);      \
  XSIMD_DECLARE_SIMD_REGISTER(SCALAR_TYPE, half_vec)

XSIMD_DECLARE_SIMD_REGISTERS(signed char);
XSIMD_DECLARE_SIMD_REGISTERS(unsigned char);
XSIMD_DECLARE_SIMD_REGISTERS(char);
XSIMD_DECLARE_SIMD_REGISTERS(short);
XSIMD_DECLARE_SIMD_REGISTERS(unsigned short);
XSIMD_DECLARE_SIMD_REGISTERS(int);
XSIMD_DECLARE_SIMD_REGISTERS(unsigned int);
XSIMD_DECLARE_SIMD_REGISTERS(long int);
XSIMD_DECLARE_SIMD_REGISTERS(unsigned long int);
XSIMD_DECLARE_SIMD_REGISTERS(long long int);
XSIMD_DECLARE_SIMD_REGISTERS(unsigned long long int);
XSIMD_DECLARE_SIMD_REGISTERS(float);
XSIMD_DECLARE_SIMD_REGISTERS(double);

// Cannot declare bool like this. Doing it manually with bool = uint8_t.
// XSIMD_DECLARE_SIMD_REGISTER(bool, generic);
template <>
struct simd_register<bool, generic> {
  using vector_type = std::array<uint8_t, generic::size()>;
  using register_type = vector_type;
  register_type data;
  operator register_type() const noexcept {
    return data;
  }
};
template <>
struct has_simd_register<bool, generic> : std::true_type {};

template <class T, class Arch>
struct get_bool_simd_register {
  using type = simd_register<T, Arch>;
};

template <class T, class Arch>
using get_bool_simd_register_t = typename get_bool_simd_register<T, Arch>::type;

namespace detail {
template <size_t S>
struct get_unsigned_type;

template <>
struct get_unsigned_type<1> {
  using type = uint8_t;
};

template <>
struct get_unsigned_type<2> {
  using type = uint16_t;
};

template <>
struct get_unsigned_type<4> {
  using type = uint32_t;
};

template <>
struct get_unsigned_type<8> {
  using type = uint64_t;
};

template <size_t S>
using get_unsigned_type_t = typename get_unsigned_type<S>::type;

template <class T, class A>
struct bool_simd_register {
  using type = simd_register<get_unsigned_type_t<sizeof(T)>, A>;
};
} // namespace detail

template <class T>
struct get_bool_simd_register<T, generic>
    : detail::bool_simd_register<T, generic> {};

} // namespace types

XSIMD_TEMPLATE
struct has_simd_register : types::has_simd_register<T, A> {};

XSIMD_TEMPLATE
struct batch;

template <typename FuncT, typename BatchT>
BatchT binary_combine(const BatchT& batch1, const BatchT& batch2) noexcept {
  BatchT result;
  FuncT func;

  for (size_t i = 0; i < BatchT::size; ++i) {
    result.data[i] = func(batch1.data[i], batch2.data[i]);
  }

  return result;
}

struct left_shift {
  template <class T, class U>
  constexpr auto operator()(T&& lhs, U&& rhs) const
      -> decltype(std::forward<T>(lhs) << std::forward<U>(rhs)) {
    return std::forward<T>(lhs) << std::forward<U>(rhs);
  }
};

struct right_shift {
  template <class T, class U>
  constexpr auto operator()(T&& lhs, U&& rhs) const
      -> decltype(std::forward<T>(lhs) >> std::forward<U>(rhs)) {
    return std::forward<T>(lhs) >> std::forward<U>(rhs);
  }
};

XSIMD_TEMPLATE struct batch_bool
    : public types::get_bool_simd_register_t<T, A> {
  static constexpr size_t size = A::size() / sizeof(T);

  using base_type = types::get_bool_simd_register_t<T, A>;
  using value_type = bool;
  using arch_type = A;
  using register_type = typename base_type::register_type;
  using batch_type = batch<T, A>;

  batch_bool() = default;
  batch_bool(bool val) noexcept {
    T initVal = val ? -1 : 0;
    std::fill_n(this->data.data(), size, initVal);
  }

  batch_bool(register_type reg) noexcept {
    this->data = reg;
  }
  batch_bool(batch_type batch) noexcept {
    // To process batch_bools the same way as done explicitly for AVX and NEON,
    // we only need to keep the most significant bit of each value.
    constexpr size_t MSB_SHIFT = (sizeof(T) * 8) - 1;
    std::transform(
        batch.data.begin(),
        batch.data.end(),
        this->data.begin(),
        [](const auto& el) {
          uint64_t bits = 0;
          std::memcpy(&bits, &el, sizeof(el));
          return bits >> MSB_SHIFT;
        });
  }
  //  template <class... Ts>
  //  batch_bool(bool val0, bool val1, Ts... vals) noexcept;

  // comparison operators
  batch_bool operator==(const batch_bool& other) const noexcept {
    return this->data == other.data;
  }
  batch_bool operator!=(const batch_bool& other) const noexcept {
    return this->data != other.data;
  }

  // logical operators
  batch_bool operator~() const noexcept {
    auto copy = *this;
    std::transform(
        copy.data.begin(), copy.data.end(), copy.data.begin(), std::bit_not());
    return copy;
  }
  batch_bool operator!() const noexcept {
    auto copy = *this;
    std::transform(
        copy.data.begin(),
        copy.data.end(),
        copy.data.begin(),
        std::logical_not());
    return copy;
  }

  batch_bool operator&(const batch_bool& other) const noexcept {
    return binary_combine<std::bit_and<>>(*this, other);
  }
  batch_bool operator|(const batch_bool& other) const noexcept {
    return binary_combine<std::bit_or<>>(*this, other);
  }
  batch_bool operator^(const batch_bool& other) const noexcept {
    return binary_combine<std::bit_xor<>>(*this, other);
  }
  batch_bool operator&&(const batch_bool& other) const noexcept {
    return binary_combine<std::logical_and<>>(*this, other);
  }
  batch_bool operator||(const batch_bool& other) const noexcept {
    return binary_combine<std::logical_or<>>(*this, other);
  }

  // update operators
  batch_bool& operator&=(const batch_bool& other) const noexcept {
    return (*this) = (*this) & other;
  }
  batch_bool& operator|=(const batch_bool& other) const noexcept {
    return (*this) = (*this) | other;
  }
  batch_bool& operator^=(const batch_bool& other) const noexcept {
    return (*this) = (*this) ^ other;
  }

  template <typename U>
  void store_aligned(U* dst) {
    for (std::size_t i = 0; i < size; ++i) {
      dst[i] = static_cast<bool>(this->data[i]);
    }
  }

  void store_unaligned(void* dst) {
    // Makes no difference here.
    store_aligned(dst);
  }

  static batch_bool load_aligned(const bool* src) {
    batch_type result;
    for (std::size_t i = 0; i < size; ++i)
      result.data[i] = src[i] ? -1 : 0;

    return result;
  }

  static batch_bool load_unaligned(const bool* src) {
    return load_aligned(src);
  }
};

template <typename T, typename A>
struct batch : public types::simd_register<T, A> {
  static constexpr size_t size = A::size() / sizeof(T);
  using arch_type = A;
  using batch_bool_type = batch_bool<T, A>;
  using register_type = typename types::simd_register<T, A>::register_type;

  batch() = default;

  batch(T val) noexcept {
    std::fill_n(this->data.data(), size, val);
  }

  explicit batch(const batch_bool_type& b) noexcept {
    this->data = reinterpret_cast<const register_type&>(b.data);
  }

  batch(register_type reg) noexcept {
    this->data = reg;
  }

  template <class... Ts>
  batch(T val0, T val1, Ts... vals) noexcept {
    static_assert(sizeof...(Ts) + 2 == size, "#args must match size of vector");
    this->data = register_type{val0, val1, vals...};
  };

  batch_bool<T, A> operator==(const batch& other) const noexcept {
    return batch_bool<T, A>(this->data == other.data);
  }
  batch_bool<T, A> operator!=(const batch& other) const noexcept {
    return batch_bool<T, A>(this->data != other.data);
  }
  batch_bool<T, A> operator>=(const batch& other) const noexcept {
    return batch_bool<T, A>(this->data >= other.data);
  }
  batch_bool<T, A> operator<=(const batch& other) const noexcept {
    return batch_bool<T, A>(this->data <= other.data);
  }
  batch_bool<T, A> operator>(const batch& other) const noexcept {
    return batch_bool<T, A>(this->data > other.data);
  }
  batch_bool<T, A> operator<(const batch& other) const noexcept {
    return batch_bool<T, A>(this->data < other.data);
  }

  batch operator^(const batch& other) const noexcept {
    return binary_combine<std::bit_xor<>>(*this, other);
  }
  batch operator&(const batch& other) const noexcept {
    return binary_combine<std::bit_and<>>(*this, other);
  }
  batch operator*(const batch& other) const noexcept {
    return binary_combine<std::multiplies<>>(*this, other);
  }
  batch operator+(const batch& other) const noexcept {
    return binary_combine<std::plus<>>(*this, other);
  }
  batch operator-(const batch& other) const noexcept {
    return binary_combine<std::minus<>>(*this, other);
  }
  batch operator<<(const batch& other) const noexcept {
    return binary_combine<left_shift>(*this, other);
  }
  batch operator>>(const batch& other) const noexcept {
    return binary_combine<right_shift>(*this, other);
  }

  T get(size_t pos) {
    return this->data[pos];
  }

  static batch broadcast(T value) {
    return batch(value);
  }

  template <typename U>
  void store_aligned(U* dst) {
    return store_unaligned(dst);
  }

  template <typename U>
  void store_unaligned(U* dst) {
    // xsimd widens or narrows during stores, so we need to as well.
    for (size_t i = 0; i < size; ++i) {
      dst[i] = this->data[i];
    }
  }

  template <typename U>
  static batch load_aligned(const U* src) {
    return load_unaligned(src);
  }

  template <typename U>
  static batch load_unaligned(const U* src) {
    batch result;

    // xsimd widens or narrows during loads, so we need to as well.
    for (size_t i = 0; i < size; ++i) {
      result.data[i] = src[i];
    }

    return result;
  }
};

template <typename T>
struct Batch64 : public batch<T, half_vec> {};
static_assert(sizeof(Batch64<int>) == 8);

///////////////////////////////
///         METHODS         ///
///////////////////////////////
XSIMD_TEMPLATE
batch<T, A> broadcast(T value) {
  return batch<T, A>::broadcast(value);
}

template <class A = default_arch, class From>
batch<From, A> load_aligned(const From* ptr) noexcept {
  return batch<From, A>::load_aligned(ptr);
}

template <class A = default_arch, class From>
batch<From, A> load_unaligned(const From* ptr) noexcept {
  return batch<From, A>::load_unaligned(ptr);
}

template <typename T, std::size_t N>
using make_sized_batch_t = batch<T, default_arch>;

} // namespace xsimd
