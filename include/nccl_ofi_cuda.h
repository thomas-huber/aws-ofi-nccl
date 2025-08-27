/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights
 * reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#ifndef NCCL_OFI_CUDA_H_
#define NCCL_OFI_CUDA_H_

/*
 * Error checking is currently just success or failure.
 */
enum {
  GPU_SUCCESS   = 0,
  GPU_ERROR     = 999  /* Match CUDA_UNKNOWN_ERROR value */
};

/* Generic GPU init (Ryan’s PR expects a gpu-generic entry point).
 * Implemented in the CUDA TU as a thin alias to nccl_net_ofi_cuda_init(). */
int nccl_net_ofi_gpu_init(void);

/*
 * Gets the CUDA device associated with the buffer.
 *
 * @param data   Pointer to CUDA buffer
 * @param dev_id Out: CUDA device id
 * @return 0 on success, -EINVAL on error
 */
int nccl_net_ofi_get_cuda_device(void *data, int *dev_id);

/* Back-compat shim (older call-sites used the _for_addr name). */
static inline int
nccl_net_ofi_get_cuda_device_for_addr(void *data, int *dev_id) {
  return nccl_net_ofi_get_cuda_device(data, dev_id);
}

/*
 * Wraps cudaDeviceFlushGPUDirectRDMAWrites() with default args.
 * Call-site expects this generic symbol name.
 *
 * @return 0 on success, non-zero on error
 */
int nccl_net_ofi_gpuFlushGPUDirectRDMAWrites(void);

/*
 * cudaGetDeviceCount()
 * @return device count on success, -1 on error
 */
int nccl_net_ofi_cuda_get_num_devices(void);

/*
 * cudaGetDevice()
 * @return active device index on success, -1 on error
 */
int nccl_net_ofi_cuda_get_active_device_idx(void);

/*
 * Query CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED
 * @return true if attribute is available and true, false otherwise
 */
bool nccl_net_ofi_cuda_have_dma_buf_attr(void);

/*
 * Query CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED and
 * CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS
 * @return true if attributes report GDR+flush supported, false otherwise
 */
bool nccl_net_ofi_cuda_have_gdr_support_attr(void);

#endif /* NCCL_OFI_CUDA_H_ */
