/*
 * Copyright (c) 2018-2025 Amazon.com, Inc. or its
 * affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_ROCM_H_
#define NCCL_OFI_ROCM_H_

/*
 * Error checking is currently just success or failure.
 */
enum {
  GPU_SUCCESS   = 0,
  GPU_ERROR     = 999
};

/* Generic GPU init (ROCm variant) */
int nccl_net_ofi_gpu_init(void);

/*
 * Gets the device associated with a GPU buffer (HIP/ROCm).
 * @return 0 on success, -EINVAL on error
 */
int nccl_net_ofi_get_cuda_device(void *data, int *dev_id);

/* back-compat shim for older call-sites */
static inline int
nccl_net_ofi_get_cuda_device_for_addr(void *data, int *dev_id) {
  return nccl_net_ofi_get_cuda_device(data, dev_id);
}

/*
 * Generic GPU-Direct-RDMA flush. ROCm does not require/offer the same
 * runtime flush; we export a stub that returns 0 so call-sites compile.
 */
int nccl_net_ofi_gpuFlushGPUDirectRDMAWrites(void);

/* Device count / active device */
int nccl_net_ofi_cuda_get_num_devices(void);
int nccl_net_ofi_cuda_get_active_device_idx(void);

/* Capability queries — return false on ROCm by default */
bool nccl_net_ofi_cuda_have_dma_buf_attr(void);
bool nccl_net_ofi_cuda_have_gdr_support_attr(void);

#endif /* NCCL_OFI_ROCM_H_ */
