/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <bit>
#include <numeric>

#include <boost/crc.hpp>

namespace facebook::velox::simd {

namespace detail {

template <typename T, typename A>
int genericToBitMask(xsimd::batch_bool<T, A> mask) {
  static_assert(mask.size <= 32);
  alignas(A::alignment()) bool tmp[mask.size];
  mask.store_aligned(tmp);
  int ans = 0;
  for (int i = 0; i < mask.size; ++i) {
    ans |= tmp[i] << i;
  }
  return ans;
}

template <typename T, typename A>
xsimd::batch_bool<T, A> fromBitMaskImpl(int mask) {
  static const auto kMemo = ({
    constexpr int N = xsimd::batch_bool<T, A>::size;
    static_assert(N <= 8);
    std::array<xsimd::batch_bool<T, A>, (1 << N)> memo;
    for (int i = 0; i < (1 << N); ++i) {
      bool tmp[N];
      for (int bit = 0; bit < N; ++bit) {
        tmp[bit] = (i & (1 << bit)) ? true : false;
      }
      memo[i] = xsimd::batch_bool<T, A>::load_unaligned(tmp);
    }
    memo;
  });
  return kMemo[mask];
}

template <typename T, typename A>
struct BitMask<T, A> {
  static constexpr int kAllSet = bits::lowMask(xsimd::batch_bool<T, A>::size);

  static int toBitMask(xsimd::batch_bool<T, A> mask, const xsimd::generic&) {
    // Generic GCC version
    //        alignas(A::alignment()) static const uint8_t kAnd[] = {
    //            1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    //        auto vAnd = xsimd::batch<T, A>::load_aligned(kAnd);
    //        auto vmask = mask & vAnd;
    //        alignas(16) std::array<uint8_t, 16> mask_arr{};
    //        std::memcpy(mask_arr.data(), &vmask, sizeof(vmask));
    //        int lo = std::accumulate(mask_arr.begin(), mask_arr.begin() + 8,
    //        0); int hi = std::accumulate(mask_arr.begin() + 8, mask_arr.end(),
    //        0); return (hi << 8) | lo;

    int result = 0;
    for (size_t i = 0; i < mask.size; ++i) {
      if (mask.data[i]) {
        result |= 1ull << i;
      }
    }
    return result;
  }

  static xsimd::batch_bool<T, A> fromBitMask(int mask, const A&) {
    return UNLIKELY(mask == kAllSet) ? xsimd::batch_bool<T, A>(true)
                                     : fromBitMaskImpl<T, A>(mask);
  }
};

} // namespace detail

template <typename A>
int32_t indicesOfSetBits(
    const uint64_t* bits,
    int32_t begin,
    int32_t end,
    int32_t* result,
    const A&) {
  if (end <= begin) {
    return 0;
  }
  int32_t row = begin & ~63;
  auto originalResult = result;
  int32_t endWord = bits::roundUp(end, 64) / 64;
  auto firstWord = begin / 64;
  for (auto wordIndex = firstWord; wordIndex < endWord; ++wordIndex) {
    uint64_t word = bits[wordIndex];
    if (!word) {
      row += 64;
      continue;
    }
    if (wordIndex == firstWord && begin != firstWord * 64) {
      word &= bits::highMask(64 - (begin - firstWord * 64));
      if (!word) {
        row += 64;
        continue;
      }
    }
    if (wordIndex == endWord - 1) {
      int32_t lastBits = end - (endWord - 1) * 64;
      if (lastBits < 64) {
        word &= bits::lowMask(lastBits);
        if (!word) {
          break;
        }
      }
    }
    if (result - originalResult < (row >> 2)) {
      do {
        *result++ = __builtin_ctzll(word) + row;
        word = word & (word - 1);
      } while (word);
      row += 64;
    } else {
      for (auto byteCnt = 0; byteCnt < 8; ++byteCnt) {
        uint8_t byte = word;
        word = word >> 8;
        if (byte) {
          using Batch = xsimd::batch<int32_t, A>;
          auto indices = byteSetBits(byte);
          if constexpr (Batch::size == 8) {
            (Batch::load_aligned(indices) + row).store_unaligned(result);
            result += __builtin_popcount(byte);
          } else {
            static_assert(Batch::size == 4);
            auto lo = byte & ((1 << 4) - 1);
            auto hi = byte >> 4;
            int pop = 0;
            if (lo) {
              (Batch::load_aligned(indices) + row).store_unaligned(result);
              pop = __builtin_popcount(lo);
              result += pop;
            }
            if (hi) {
              (Batch::load_unaligned(indices + pop) + row)
                  .store_unaligned(result);
              result += __builtin_popcount(hi);
            }
          }
        }
        row += 8;
      }
    }
  }
  return result - originalResult;
}

template <typename T, typename A>
xsimd::batch_bool<T, A> leadingMask(int n, const A&) {
  constexpr int N = xsimd::batch_bool<T, A>::size;
  static const auto kMemo = ({
    std::array<xsimd::batch_bool<T, A>, N> memo;
    bool tmp[N]{};
    for (int i = 0; i < N; ++i) {
      memo[i] = xsimd::batch_bool<T, A>::load_unaligned(tmp);
      tmp[i] = true;
    }
    memo;
  });
  return LIKELY(n >= N) ? xsimd::batch_bool<T, A>(true) : kMemo[n];
}

namespace detail {

template <typename T, typename A>
struct CopyWord {
  static void apply(void* to, const void* from) {
    *reinterpret_cast<T*>(to) = *reinterpret_cast<const T*>(from);
  }
};

template <typename A>
struct CopyWord<xsimd::batch<int8_t, A>, A> {
  static void apply(void* to, const void* from) {
    xsimd::batch<int8_t, A>::load_unaligned(
        reinterpret_cast<const int8_t*>(from))
        .store_unaligned(reinterpret_cast<int8_t*>(to));
  }
};

// Copies one element of T and advances 'to', 'from', and 'bytes' by
// sizeof(T). Returns false if 'bytes' went to 0.
template <typename T, typename A>
inline bool copyNextWord(void*& to, const void*& from, int32_t& bytes) {
  if (bytes >= sizeof(T)) {
    CopyWord<T, A>::apply(to, from);
    bytes -= sizeof(T);
    if (!bytes) {
      return false;
    }
    from = addBytes(from, sizeof(T));
    to = addBytes(to, sizeof(T));
    return true;
  }
  return true;
}
} // namespace detail

template <typename A>
void memcpy(void* to, const void* from, int32_t bytes, const A& arch) {
  while (bytes >= batchByteSize(arch)) {
    if (!detail::copyNextWord<xsimd::batch<int8_t, A>, A>(to, from, bytes)) {
      return;
    }
  }
  while (bytes >= sizeof(int64_t)) {
    if (!detail::copyNextWord<int64_t, A>(to, from, bytes)) {
      return;
    }
  }
  if (!detail::copyNextWord<int32_t, A>(to, from, bytes)) {
    return;
  }
  if (!detail::copyNextWord<int16_t, A>(to, from, bytes)) {
    return;
  }
  detail::copyNextWord<int8_t, A>(to, from, bytes);
}

namespace detail {

template <typename T, typename A>
struct SetWord {
  static void apply(void* to, T data) {
    *reinterpret_cast<T*>(to) = data;
  }
};

template <typename A>
struct SetWord<xsimd::batch<int8_t, A>, A> {
  static void apply(void* to, xsimd::batch<int8_t, A> data) {
    data.store_unaligned(reinterpret_cast<int8_t*>(to));
  }
};

template <typename T, typename A>
inline bool setNextWord(void*& to, T data, int32_t& bytes, const A&) {
  if (bytes >= sizeof(T)) {
    SetWord<T, A>::apply(to, data);
    bytes -= sizeof(T);
    if (!bytes) {
      return false;
    }
    to = addBytes(to, sizeof(T));
    return true;
  }
  return true;
}

} // namespace detail

template <typename A>
void memset(void* to, char data, int32_t bytes, const A& arch) {
  auto v = xsimd::batch<int8_t, A>::broadcast(data);
  while (bytes >= batchByteSize(arch)) {
    if (!detail::setNextWord(to, v, bytes, arch)) {
      return;
    }
  }
  int64_t data64 = *reinterpret_cast<int64_t*>(&v);
  while (bytes >= sizeof(int64_t)) {
    if (!detail::setNextWord<int64_t>(to, data64, bytes, arch)) {
      return;
    }
  }
  if (!detail::setNextWord<int32_t>(to, data64, bytes, arch)) {
    return;
  }
  if (!detail::setNextWord<int16_t>(to, data64, bytes, arch)) {
    return;
  }
  detail::setNextWord<int8_t>(to, data64, bytes, arch);
}

namespace detail {

template <typename T, typename A>
struct HalfBatchImpl<T, A, std::enable_if_t<A::size() == 16>> {
  using Type = Batch64<T>;
};

template <typename T, typename A>
struct HalfBatchImpl<T, A, std::enable_if_t<A::size() == 32>> {
  using Type = xsimd::batch<T, xsimd::generic16>;
};

template <typename T, typename A>
struct HalfBatchImpl<T, A, std::enable_if_t<A::size() == 64>> {
  using Type = xsimd::batch<T, xsimd::generic32>;
};

// Works reasonably well: https://godbolt.org/z/b8os9s8Er
template <typename T, typename A, int kScale, typename IndexType>
xsimd::batch<T, A> genericGather(
    const T* base,
    xsimd::batch<IndexType, A> indices) {
  constexpr int N = xsimd::batch<T, A>::size;

  xsimd::batch<T, A> res;
  if constexpr (kScale == sizeof(T)) {
    for (int i = 0; i < N; ++i) {
      res.data[i] = *(base + indices.data[i]);
    }
  } else {
    const auto* bytes = reinterpret_cast<const char*>(base);
    indices = indices * kScale;
    for (int i = 0; i < N; ++i) {
      res.data[i] = *reinterpret_cast<const T*>(bytes + indices.data[i]);
    }
  }

  return res;
}

template <typename T, typename A, int kScale, typename IndexType, typename IdxA>
xsimd::batch<T, A> genericMaskGather(
    xsimd::batch<T, A> src,
    xsimd::batch_bool<T, A> mask,
    const T* base,
    xsimd::batch<IndexType, IdxA> vindex) {
  constexpr int N = xsimd::batch<T, A>::size;

  alignas(A::alignment()) T dst[N];
  alignas(A::alignment()) T sr[N];
  alignas(A::alignment()) bool ma[N];

  src.store_aligned(sr);
  mask.store_aligned(ma);
  auto bytes = reinterpret_cast<const char*>(base);

  for (int i = 0; i < N; ++i) {
    if (ma[i]) {
      dst[i] = *reinterpret_cast<const T*>(bytes + vindex.data[i] * kScale);
    } else {
      dst[i] = sr[i];
    }
  }
  return xsimd::load_aligned(dst);
}

template <typename T, typename A>
struct Gather<T, int32_t, A, 2> {
  using VIndexType = xsimd::batch<int32_t, A>;

  // Load 8 indices only.
  static VIndexType loadIndices(const int32_t* indices, const A& arch) {
    return Gather<int32_t, int32_t, A>::loadIndices(indices, arch);
  }
};

template <typename T, typename A>
struct Gather<T, int32_t, A, 4> {
  using VIndexType = xsimd::batch<int32_t, A>;

  static VIndexType loadIndices(const int32_t* indices, const xsimd::generic&) {
    return xsimd::load_unaligned<A>(indices);
  }

  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, VIndexType vindex, const xsimd::generic&) {
    return genericGather<T, A, kScale>(base, vindex);
  }

  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, const int32_t* indices, const xsimd::generic&) {
    auto vindex = VIndexType::load_unaligned(indices);
    return apply<kScale>(base, vindex, A{});
  }

  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> src,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      const int32_t* indices,
      const xsimd::generic&) {
    return genericMaskGather<T, A, kScale>(
        src, mask, base, loadIndices(indices, A{}));
  }

  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> src,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      VIndexType vindex,
      const xsimd::generic&) {
    return genericMaskGather<T, A, kScale>(src, mask, base, vindex);
  }
};

template <typename T, typename A>
struct Gather<T, int32_t, A, 8> {
  static HalfBatch<int32_t> loadIndices(
      const int32_t* indices,
      const xsimd::generic&) {
    return HalfBatch<int32_t>::load_unaligned(indices);
  }

  using VIndexType = xsimd::batch<int32_t, A>;

  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, const int32_t* indices, const xsimd::generic&) {
    auto vindex = VIndexType::load_unaligned(indices);
    return apply<kScale>(base, vindex, A{});
  }

  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, VIndexType vindex, const xsimd::generic&) {
    return genericGather<T, A, kScale>(base, vindex);
  }

  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> src,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      const int32_t* indices,
      const xsimd::generic&) {
    return genericMaskGather<T, A, kScale>(
        src, mask, base, loadIndices(indices, A{}));
  }
};

template <typename T, typename A>
struct Gather<T, int64_t, A, 8> {
  using VIndexType = xsimd::batch<int64_t, A>;

  static VIndexType loadIndices(const int64_t* indices, const xsimd::generic&) {
    return xsimd::load_unaligned<A>(indices);
  }

  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, const int64_t* indices, const xsimd::generic&) {
    return genericGather<T, A, kScale>(base, loadIndices(indices, A{}));
  }

  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> src,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      const int64_t* indices,
      const xsimd::generic&) {
    return genericMaskGather<T, A, kScale>(
        src, mask, base, loadIndices(indices, A{}));
  }

  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> src,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      VIndexType vindex,
      const xsimd::generic&) {
    return genericMaskGather<T, A, kScale>(src, mask, base, vindex);
  }
};

// Concatenates the low 16 bits of each lane in 'x' and 'y' and
// returns the result as 16x16 bits.
template <typename A>
xsimd::batch<int16_t, A> pack32(
    xsimd::batch<int32_t, A> x,
    xsimd::batch<int32_t, A> y,
    const xsimd::generic&) {
  alignas(2 * x.size) std::array<uint16_t, 2 * x.size> out_data;
  for (size_t i = 0; i < x.size; ++i) {
    out_data[i] = static_cast<uint32_t>(x.get(i));
  }
  for (size_t i = 0; i < y.size; ++i) {
    out_data[x.size + i] = static_cast<uint32_t>(y.get(i));
  }

  return xsimd::batch<int16_t, A>::load_aligned(out_data.data());
}

template <typename T, typename A>
xsimd::batch<T, A> genericPermute(xsimd::batch<T, A> data, const int32_t* idx) {
  constexpr int N = xsimd::batch<T, A>::size;
   alignas(A::alignment()) T src[N];
   alignas(A::alignment()) T dst[N];
   data.store_aligned(src);
   for (int i = 0; i < N; ++i) {
     dst[i] = idx[i] < 0 ? 0 : src[idx[i]];
   }
   return xsimd::load_aligned<A>(dst);
}

template <typename T, typename A>
xsimd::batch<T, A> genericPermute(
    xsimd::batch<T, A> data,
    xsimd::batch<int32_t, A> idx) {
  static_assert(data.size >= idx.size);
  alignas(A::alignment()) int32_t pos[idx.size];
  idx.store_aligned(pos);
  return genericPermute(data, pos);
}

template <typename T>
Batch64<T> genericPermute(Batch64<T> data, Batch64<int32_t> idx) {
  static_assert(data.size >= idx.size);
  Batch64<T> ans;
  for (int i = 0; i < idx.size; ++i) {
    ans.data[i] = data.data[idx.data[i]];
  }
  return ans;
}

template <typename T, typename A, size_t kSizeT = sizeof(T)>
struct Permute;

template <typename T, typename A>
struct Permute<T, A, 4> {
  static xsimd::batch<T, A> apply(
      xsimd::batch<T, A> data,
      xsimd::batch<int32_t, A> idx,
      const xsimd::generic&) {
    return genericPermute(data, idx);
  }

  static HalfBatch<T, A> apply(
      HalfBatch<T, A> data,
      HalfBatch<int32_t, A> idx,
      const xsimd::generic&) {
    return genericPermute(data, idx);
  }
};

} // namespace detail

template <int kScale, typename A>
xsimd::batch<int16_t, A> gather(
    const int16_t* base,
    const int32_t* indices,
    int numIndices,
    const A& arch) {
  auto first = maskGather<int32_t, int32_t, kScale>(
      xsimd::batch<int32_t, A>::broadcast(0),
      leadingMask<int32_t>(numIndices, arch),
      reinterpret_cast<const int32_t*>(base),
      indices,
      arch);
  xsimd::batch<int32_t, A> second;
  constexpr int kIndicesBatchSize = xsimd::batch<int32_t, A>::size;
  if (numIndices > kIndicesBatchSize) {
    second = maskGather<int32_t, int32_t, kScale>(
        xsimd::batch<int32_t, A>::broadcast(0),
        leadingMask<int32_t>(numIndices - kIndicesBatchSize, arch),
        reinterpret_cast<const int32_t*>(base),
        indices + kIndicesBatchSize,
        arch);
  } else {
    second = xsimd::batch<int32_t, A>::broadcast(0);
  }
  return detail::pack32(first, second, arch);
}

namespace detail {

template <typename A>
uint8_t gather8BitsImpl(
    const void* bits,
    xsimd::batch<int32_t, A> vindex,
    int32_t numIndices,
    const xsimd::generic&) {
  alignas(A::alignment()) int32_t indices[vindex.size];
  vindex.store_aligned(indices);
  auto base = reinterpret_cast<const char*>(bits);
  uint8_t ans = 0;
  for (int i = 0, n = std::min<int>(vindex.size, numIndices); i < n; ++i) {
    bits::setBit(&ans, i, bits::isBitSet(base, indices[i]));
  }
  return ans;
}

} // namespace detail

template <typename A>
uint8_t gather8Bits(
    const void* bits,
    xsimd::batch<int32_t, A> vindex,
    int32_t numIndices,
    const A& arch) {
  return detail::gather8BitsImpl(bits, vindex, numIndices, arch);
}

namespace detail {

template <typename TargetT, typename SourceT, typename A>
struct GetHalf {
  static_assert(std::is_same_v<SourceT, int32_t>, "unexpected?");

  template <bool kSecond>
  static xsimd::batch<TargetT, A> apply(
      xsimd::batch<SourceT, A> data,
      const xsimd::generic&) {
    constexpr size_t offset = kSecond * data.size / 2;

    // original AVX code uses cvtepi with sign-extension if TargetT is signed,
    // otherwise cvtepu with zero-extension.

    alignas(data.size) std::array<TargetT, data.size / 2> result_data;
    for (size_t i = 0; i < data.size / 2; ++i) {
      if constexpr (std::is_unsigned_v<TargetT>) {
        result_data[i] =
            static_cast<std::make_unsigned_t<SourceT>>(data.get(offset + i));
      } else {
        result_data[i] = data.get(offset + i);
      }
    }

    return xsimd::batch<TargetT, A>::load_aligned(result_data.data());
  }
};

} // namespace detail

namespace detail {

// Indices to use in 8x32 bit permute for extracting words from 4x64
// bits.  The entry at 5 (bits 0 and 2 set) is {0, 1, 4, 5, 4, 5, 6,
// 7}, meaning 64 bit words at 0 and 2 are moved in front (to 0, 1).
extern int32_t permute4x64Indices[16][8];

template <typename T, typename A>
struct Filter<T, A, 2> {
  static xsimd::batch<T, A>
  apply(xsimd::batch<T, A> data, int mask, const xsimd::generic&) {
    xsimd::batch<T, A> result(0);
    auto* write_ptr = result.data.begin();
    for (int i = 0; i < data.size; ++i) {
      if (mask & (1ull << i)) {
        *write_ptr++ = data.data[i];
      }
    }

    return result;
  }
};

template <typename T, typename A>
struct Filter<T, A, 4> {
  static xsimd::batch<T, A>
  apply(xsimd::batch<T, A> data, int mask, const A& arch) {
    auto vindex = xsimd::batch<int32_t, A>::load_aligned(byteSetBits[mask]);
    return Permute<T, A>::apply(data, vindex, arch);
  }

  static HalfBatch<T, A> apply(HalfBatch<T, A> data, int mask, const A& arch) {
    auto vindex = HalfBatch<int32_t, A>::load_aligned(byteSetBits[mask]);
    return Permute<T, A>::apply(data, vindex, arch);
  }
};

template <typename T, typename A>
struct Filter<T, A, 8> {
  static xsimd::batch<T, A>
  apply(xsimd::batch<T, A> data, int mask, const xsimd::generic&) {
    return genericPermute(data, byteSetBits[mask]);
  }
};

template <typename A>
struct Crc32<uint64_t, A> {
  static uint32_t
  apply(uint32_t checksum, uint64_t value, const xsimd::generic&) {
    boost::crc_32_type result(checksum);
    result.process_bytes(&value, sizeof(value));
    return result();
  }
};

} // namespace detail

template <typename T, typename A>
xsimd::batch<T, A> iota(const A&) {
  static const auto kMemo = ({
    constexpr int N = xsimd::batch<T, A>::size;
    T tmp[N];
    std::iota(tmp, tmp + N, 0);
    xsimd::load_unaligned(tmp);
  });
  return kMemo;
}

namespace detail {

template <typename T, typename U, typename A>
struct ReinterpretBatch {
  static xsimd::batch<T, A> apply(
      xsimd::batch<U, A> data,
      const xsimd::generic&) {
    return xsimd::batch<T, A>::load_aligned(data.data.data());
  }
};

template <typename T, typename A>
struct ReinterpretBatch<T, T, A> {
  static xsimd::batch<T, A> apply(xsimd::batch<T, A> data, const A&) {
    return data;
  }
};

} // namespace detail

template <typename T, typename U, typename A>
xsimd::batch<T, A> reinterpretBatch(xsimd::batch<U, A> data, const A& arch) {
  return detail::ReinterpretBatch<T, U, A>::apply(data, arch);
}

} // namespace facebook::velox::simd
