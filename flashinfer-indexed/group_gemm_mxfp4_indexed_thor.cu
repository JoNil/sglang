/*
 * Thor-specialized indexed MXFP4 group GEMM binding.
 *
 * DeepSeek-V4 always supplies MXFP8 E4M3 activations, packed E2M1 weights,
 * E8M0 block scales, BF16 output, and SwapAB=true. Keeping those types fixed
 * avoids compiling every generic SM100 group-GEMM permutation at startup.
 */
#include <flashinfer/cutlass_utils.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

#define DISPATCH_MMA_SM(mma_sm, MMA_SM, ...)       \
  [&]() -> bool {                                  \
    if (mma_sm == 1) {                             \
      constexpr int MMA_SM = 1;                    \
      return __VA_ARGS__();                        \
    } else if (mma_sm == 2) {                      \
      constexpr int MMA_SM = 2;                    \
      return __VA_ARGS__();                        \
    }                                              \
    TVM_FFI_ICHECK(false) << "Unsupported MMA SM"; \
    return false;                                  \
  }()

#define DISPATCH_TILE_N(tile_n, TILE_N, ...)       \
  [&]() -> bool {                                  \
    if (tile_n == 64) {                            \
      constexpr int TILE_N = 64;                   \
      return __VA_ARGS__();                        \
    } else if (tile_n == 128) {                    \
      constexpr int TILE_N = 128;                  \
      return __VA_ARGS__();                        \
    } else if (tile_n == 192) {                    \
      constexpr int TILE_N = 192;                  \
      return __VA_ARGS__();                        \
    } else if (tile_n == 256) {                    \
      constexpr int TILE_N = 256;                  \
      return __VA_ARGS__();                        \
    }                                              \
    TVM_FFI_ICHECK(false) << "Unsupported TILE N"; \
    return false;                                  \
  }()

#define DISPATCH_TILE_K(tile_k, TILE_K, ...)       \
  [&]() -> bool {                                  \
    if (tile_k == 128) {                           \
      constexpr int TILE_K = 128;                  \
      return __VA_ARGS__();                        \
    } else if (tile_k == 256) {                    \
      constexpr int TILE_K = 256;                  \
      return __VA_ARGS__();                        \
    }                                              \
    TVM_FFI_ICHECK(false) << "Unsupported TILE K"; \
    return false;                                  \
  }()

namespace flashinfer {
namespace group_gemm {

template <int TileM, int TileN, int TileK, int MmaSM, bool SwapAB, typename DTypeInA,
          typename DTypeInB, typename DTypeSFA, typename DTypeSFB, typename DTypeOut>
cudaError_t CutlassMXFP4GroupwiseScaledGroupGEMMSM100(
    void* int_buffer, size_t int_buffer_size_in_bytes, void* float_buffer,
    size_t float_buffer_size_in_bytes, DTypeInA* A, DTypeInB* B, DTypeSFA* SFA, DTypeSFB* SFB,
    DTypeOut* D, int* m_indptr, int* expert_ids, int n, int k, int num_groups,
    cudaStream_t stream);

}  // namespace group_gemm
}  // namespace flashinfer

void CutlassGroupGemmMXFP4GroupwiseScaledIndexedThor(
    TensorView int_workspace_buffer, TensorView float_workspace_buffer, TensorView A, TensorView B,
    TensorView SFA, TensorView SFB, TensorView D, TensorView m_indptr, TensorView expert_ids,
    int64_t n, int64_t k, int64_t mma_sm, int64_t tile_n, int64_t tile_k) {
  ffi::CUDADeviceGuard device_guard(float_workspace_buffer.device().device_id);
  auto stream = get_stream(A.device());
  int num_groups = m_indptr.size(0) - 1;
  DISPATCH_MMA_SM(mma_sm, MMA_SM, [&] {
    return DISPATCH_TILE_N(tile_n, TILE_N, [&] {
      return DISPATCH_TILE_K(tile_k, TILE_K, [&] {
        auto status = flashinfer::group_gemm::CutlassMXFP4GroupwiseScaledGroupGEMMSM100<
            128, TILE_N, TILE_K, MMA_SM, true, cutlass::float_e4m3_t, cutlass::float_e2m1_t,
            cutlass::float_ue8m0_t, cutlass::float_ue8m0_t, cutlass::bfloat16_t>(
            static_cast<int*>(int_workspace_buffer.data_ptr()),
            get_element_size(int_workspace_buffer) * int_workspace_buffer.size(0),
            static_cast<float*>(float_workspace_buffer.data_ptr()),
            get_element_size(float_workspace_buffer) * float_workspace_buffer.size(0),
            static_cast<cutlass::float_e4m3_t*>(A.data_ptr()),
            static_cast<cutlass::float_e2m1_t*>(B.data_ptr()),
            static_cast<cutlass::float_ue8m0_t*>(SFA.data_ptr()),
            static_cast<cutlass::float_ue8m0_t*>(SFB.data_ptr()),
            static_cast<cutlass::bfloat16_t*>(D.data_ptr()),
            static_cast<int*>(m_indptr.data_ptr()), static_cast<int*>(expert_ids.data_ptr()), n, k,
            num_groups, stream);
        return status == cudaSuccess;
      });
    });
  });
}
