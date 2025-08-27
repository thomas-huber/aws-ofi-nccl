# Copyright (c) 2018-2025 Amazon.com, Inc. or its affiliates.

AM_CPPFLAGS = -I$(abs_top_srcdir)/include
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party
# Use the device-mapped interface dir from configure.ac (you map rocm->cuda there)
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/nccl/$(DEVICE_INTERFACE)/include
AM_CPPFLAGS += $(CUDA_CPPFLAGS) $(ROCM_CPPFLAGS)
AM_CPPFLAGS += -DXML_DIR=\"${pkgdatadir}/xml\"

# ---------------- Common sources (C++) ----------------
sources_common = \
  cm/nccl_ofi_cm.cpp \
  cm/nccl_ofi_cm_resources.cpp \
  cm/nccl_ofi_cm_reqs.cpp \
  nccl_ofi_api.cpp \
  nccl_ofi_assert.cpp \
  nccl_ofi_compat.cpp \
  nccl_ofi_net.cpp \
  nccl_ofi_sendrecv.cpp \
  nccl_ofi_system.cpp \
  nccl_ofi_rdma.cpp \
  nccl_ofi_scheduler.cpp \
  nccl_ofi_topo.cpp \
  nccl_ofi_mr.cpp \
  nccl_ofi_msgbuff.cpp \
  nccl_ofi_nccl_compat.cpp \
  nccl_ofi_freelist.cpp \
  nccl_ofi_idpool.cpp \
  nccl_ofi_ofiutils.cpp \
  nccl_ofi_pthread.cpp \
  nccl_ofi_dmabuf.cpp \
  nccl_ofi_ep_addr_list.cpp \
  nccl_ofi_param.cpp \
  tracepoint.cpp

# ---------------- Device-specific sources ----------------
sources_device =

if ENABLE_NEURON
sources_device += nccl_ofi_interface_neuron.cpp
endif

if HAVE_ROCM
sources_device += nccl_ofi_rocm.cpp nccl_ofi_interface_nvidia.cpp
endif

if HAVE_CUDA
sources_device += nccl_ofi_cuda.cpp nccl_ofi_interface_nvidia.cpp
endif

# ---------------- Internal convenience lib ----------------
noinst_LTLIBRARIES = libinternal_plugin.la
libinternal_plugin_la_SOURCES = $(sources_common) $(sources_device)
libinternal_plugin_la_LDFLAGS = -static $(CUDA_LDFLAGS) $(ROCM_LDFLAGS)
libinternal_plugin_la_LIBADD  = $(CUDA_LIBS) $(ROCM_LIBS)

# ---------------- Public plugin .so ----------------
if ENABLE_NEURON
lib_LTLIBRARIES = libnccom-net.la
libnccom_net_la_SOURCES =
libnccom_net_la_LIBADD = libinternal_plugin.la
libnccom_net_la_LIBTOOLFLAGS = --tag=CXX
libnccom_net_la_LDFLAGS = -module -avoid-version
else
lib_LTLIBRARIES = libnccl-net-ofi.la
libnccl_net_ofi_la_SOURCES =
libnccl_net_ofi_la_LIBADD = libinternal_plugin.la
libnccl_net_ofi_la_LIBTOOLFLAGS = --tag=CXX
libnccl_net_ofi_la_LDFLAGS = -module -avoid-version
endif

