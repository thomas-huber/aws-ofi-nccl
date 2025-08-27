/*
 * Copyright (c) 2018-2025 Amazon.com,
 * Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <errno.h>
#include <hip/hip_runtime_api.h>

#include "nccl_ofi.h"
#include "nccl_ofi_rocm.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_param.h"


/* ---- generic init expected by the gpu-generic layer ---- */
int nccl_net_ofi_gpu_init(void)
{
  int drv = -1, rt = -1;
  hipError_t e;

  e = hipDriverGetVersion(&drv);
  if (e != hipSuccess) {
    NCCL_OFI_WARN("Failed to query ROCm driver version.");
    return -EINVAL;
  }
  e = hipRuntimeGetVersion(&rt);
  if (e != hipSuccess) {
    NCCL_OFI_WARN("Failed to query ROCm runtime version.");
    return -EINVAL;
  }

  NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
                "Using ROCm driver version %d with runtime %d", drv, rt);

  /* No CUDA-style GDR flush on ROCm; leave cuda_flush=false in init path. */
  return 0;
}

/* ---- generic GPU flush symbol used at call-sites ----
 * ROCm has no cudaDeviceFlushGPUDirectRDMAWrites equivalent.
 * Export a stub that returns success so callers that guard on cuda_flush
 * still compile and (effectively) no-op here.
 */
int nccl_net_ofi_gpuFlushGPUDirectRDMAWrites(void)
{
  return 0;
}

/* Device count */
int nccl_net_ofi_cuda_get_num_devices(void)
{
  int count = -1;
  hipError_t e = hipGetDeviceCount(&count);
  return (e == hipSuccess) ? count : -1;
}

/* Active device */
int nccl_net_ofi_cuda_get_active_device_idx(void)
{
  int idx = -1;
  hipError_t e = hipGetDevice(&idx);
  return (e == hipSuccess) ? idx : -1;
}

/* Resolve device owning a pointer */
int nccl_net_ofi_get_cuda_device(void *data, int *dev_id)
{
  hipPointerAttribute_t attrs{};
  hipError_t e = hipPointerGetAttributes(&attrs, data);
  if (e != hipSuccess) {
    *dev_id = -1;
    return -EINVAL;
  }

  if (attrs.type == hipMemoryTypeDevice || attrs.isManaged) {
    *dev_id = attrs.device;
    return 0;
  }

  NCCL_OFI_WARN("Invalid (non-device) pointer supplied to HIP device resolver");
  *dev_id = -1;
  return -EINVAL;
}

/* Capabilities: default to false on ROCm */
bool nccl_net_ofi_cuda_have_dma_buf_attr(void) { return false; }
bool nccl_net_ofi_cuda_have_gdr_support_attr(void) { return false; }
