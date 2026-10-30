#include "Driver/GPU/HggcApi.h"
#include "Driver/Dispatch.h"

namespace proton {

namespace hggc {

struct ExternLibHggc : public ExternLibBase {
  using RetType = HGresult;
  static constexpr const char *name = "libhggc.so";
  static constexpr RetType success = HGGC_SUCCESS;
  static inline void *lib = nullptr;
};

DEFINE_DISPATCH(ExternLibHggc, init, hgInit, unsigned int)

DEFINE_DISPATCH(ExternLibHggc, ctxSynchronize, hgCtxSynchronize)

DEFINE_DISPATCH(ExternLibHggc, ctxGetCurrent, hgCtxGetCurrent, HGcontext *)

DEFINE_DISPATCH(ExternLibHggc, ctxGetDevice, hgCtxGetDevice, HGdevice *)

DEFINE_DISPATCH(ExternLibHggc, ctxGetStreamPriorityRange,
                hgCtxGetStreamPriorityRange, int *, int *)

DEFINE_DISPATCH(ExternLibHggc, deviceGet, hgDeviceGet, HGdevice *, int)

DEFINE_DISPATCH(ExternLibHggc, deviceGetAttribute, hgDeviceGetAttribute, int *,
                HGdevice_attribute, HGdevice)

DEFINE_DISPATCH(ExternLibHggc, streamCreateWithPriority,
                hgStreamCreateWithPriority, HGstream *, unsigned int, int)

DEFINE_DISPATCH(ExternLibHggc, streamSynchronize, hgStreamSynchronize, HGstream)

DEFINE_DISPATCH(ExternLibHggc, streamDestroy, hgStreamDestroy_v2, HGstream)

DEFINE_DISPATCH(ExternLibHggc, memcpyDToHAsync, hgMemcpyDtoHAsync_v2, void *,
                HGdeviceptr, size_t, HGstream)

DEFINE_DISPATCH(ExternLibHggc, memsetD32Async, hgMemsetD32Async, HGdeviceptr,
                unsigned int, size_t, HGstream)

DEFINE_DISPATCH(ExternLibHggc, memAlloc, hgMemAlloc_v2, HGdeviceptr *, size_t)

DEFINE_DISPATCH(ExternLibHggc, memFree, hgMemFree_v2, HGdeviceptr)

DEFINE_DISPATCH(ExternLibHggc, memAllocHost, hgMemAllocHost_v2, void **, size_t)

DEFINE_DISPATCH(ExternLibHggc, memHostAlloc, hgMemHostAlloc, void **, size_t,
                unsigned int)

DEFINE_DISPATCH(ExternLibHggc, memHostGetDevicePointer,
                hgMemHostGetDevicePointer_v2, HGdeviceptr *, void *, unsigned int)

DEFINE_DISPATCH(ExternLibHggc, memFreeHost, hgMemFreeHost, void *)

DEFINE_DISPATCH(ExternLibHggc, launchKernel, hgLaunchKernel, HGfunction,
                unsigned int, unsigned int, unsigned int, unsigned int,
                unsigned int, unsigned int, unsigned int, HGstream, void **,
                void **)

Device getDevice(uint64_t index) {
  HGdevice device;
  hggc::deviceGet<true>(&device, index);
  int clockRate;
  hggc::deviceGetAttribute<true>(&clockRate, HG_DEVICE_ATTRIBUTE_CLOCK_RATE,
                               device);
  int memoryClockRate;
  hggc::deviceGetAttribute<true>(&memoryClockRate,
                               HG_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device);
  int busWidth;
  hggc::deviceGetAttribute<true>(
      &busWidth, HG_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device);
  int numSms;
  hggc::deviceGetAttribute<true>(
      &numSms, HG_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
  int major;
  hggc::deviceGetAttribute<true>(
      &major, HG_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  int minor;
  hggc::deviceGetAttribute<true>(
      &minor, HG_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  std::string arch = std::to_string(major * 10 + minor);

  return Device(DeviceType::PPU, index, clockRate, memoryClockRate, busWidth,
                numSms, arch);
}

} // namespace hggc

} // namespace proton
