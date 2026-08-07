/*
 * Copyright (c) 2025 by FlashInfer team.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "tvm_ffi_utils.h"

void CutlassGroupGemmMXFP4GroupwiseScaledIndexedThor(
    TensorView int_workspace_buffer, TensorView float_workspace_buffer, TensorView A, TensorView B,
    TensorView SFA, TensorView SFB, TensorView D, TensorView m_indptr, TensorView expert_ids,
    int64_t n, int64_t k, int64_t mma_sm, int64_t tile_n, int64_t tile_k);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(group_gemm_mxfp4_nt_groupwise_indexed,
                              CutlassGroupGemmMXFP4GroupwiseScaledIndexedThor);
