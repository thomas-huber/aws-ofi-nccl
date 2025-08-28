# Copyright (c) 2018-2025 Amazon.com, Inc. or its affiliates.

AM_CPPFLAGS = -I$(abs_top_srcdir)/include
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party
# Always expose the NCCL/RCCL net headers we vendor (this path exists in your tree)
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/nccl/cuda/include
# Device-specific flags injected by configure
AM_CPPFLAGS += $(CUDA_CPPFLAGS) $(ROCM_CPPFLAGS)
# XML files packaged with the plugin
AM_CPPFLAGS += -DXML_DIR=\"${pkgdatadir}/xml\"

# --------- common sources (C++) ----------
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

# --------- device-specific sources ----------
# Avoid Automake "multiply defined" warnings by defining once and appending in conditionals.
sources_device =

if ENABLE_NEURON
sources_device += \
  nccl_ofi_interface_neuron.cpp
endif

if HAVE_ROCM
sources_device += \
  nccl_ofi_rocm.cpp \
  nccl_ofi_interface_nvidia.cpp
endif

if HAVE_CUDA
sources_device += \
  nccl_ofi_cuda.cpp \
  nccl_ofi_interface_nvidia.cpp
endif

# Internal static convenience library used by the runtime .so
noinst_LTLIBRARIES = libinternal_plugin.la
libinternal_plugin_la_SOURCES = $(sources_common) $(sources_device)
libinternal_plugin_la_LDFLAGS = -static $(CUDA_LDFLAGS) $(ROCM_LDFLAGS)
libinternal_plugin_la_LIBADD  = $(CUDA_LIBS) $(ROCM_LIBS)

# Public plugin .so (NCCL or Neuron)
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

# ---- Install-time symlinks so NCCL/RCCL find the plugin by default ----
# We always install libnccl-net-ofi.so above. Optionally add well-known names:
# - libnccl-net.so  (if ENABLE_NCCL_NET_SYMLINK)
# - librccl-net.so  (when building with ROCm)
symlink_files =
if ENABLE_NCCL_NET_SYMLINK
symlink_files += libnccl-net.so
endif
if HAVE_ROCM
symlink_files += librccl-net.so
endif

install_plugin_symlinks = { \
  test -z "$$files" \
    || { test ! -d "$$dir" && test ! -f "$$dir" && test ! -r "$$dir"; } \
    || { for file in $$files ; do \
         echo " ( cd '$$dir' && rm -f $$file && $(LN_S) libnccl-net-ofi.so $$file )"; \
         $(am__cd) "$$dir" && rm -f $$file && $(LN_S) libnccl-net-ofi.so $$file ; \
       done ; } \
  }

install-exec-hook:
	@files="$(symlink_files)" ; dir='$(DESTDIR)$(libdir)' ; $(install_plugin_symlinks)

uninstall-local:
	@files="$(symlink_files)" ; dir='$(DESTDIR)$(libdir)' ; $(am__uninstall_files_from_dir)
