#ifndef PROTON_DRIVER_GPU_HGGC_API_H_
#define PROTON_DRIVER_GPU_HGGC_API_H_

#include "Device.h"
#include "hggc.h"
#include <cstddef>

namespace proton {

namespace hggc {

template <bool CheckSuccess> HGresult init(unsigned int flags);

template <bool CheckSuccess> HGresult ctxSynchronize();

template <bool CheckSuccess> HGresult ctxGetCurrent(HGcontext *pctx);

template <bool CheckSuccess> HGresult ctxGetDevice(HGdevice *device);

template <bool CheckSuccess>
HGresult ctxGetStreamPriorityRange(int *leastPriority, int *greatestPriority);

template <bool CheckSuccess>
HGresult deviceGetAttribute(int *pi, HGdevice_attribute attrib, HGdevice dev);

template <bool CheckSuccess> HGresult deviceGet(HGdevice *device, int ordinal);

template <bool CheckSuccess>
HGresult streamCreateWithPriority(HGstream *pStream, unsigned int flags,
                                  int priority);

template <bool CheckSuccess> HGresult streamSynchronize(HGstream stream);

template <bool CheckSuccess> HGresult streamDestroy(HGstream stream);

template <bool CheckSuccess>
HGresult memcpyDToHAsync(void *dst, HGdeviceptr src, size_t count,
                         HGstream stream);

template <bool CheckSuccess>
HGresult memsetD32Async(HGdeviceptr dst, unsigned int ui, size_t N,
                        HGstream stream);

template <bool CheckSuccess>
HGresult memAlloc(HGdeviceptr *dptr, size_t bytesize);

template <bool CheckSuccess> HGresult memFree(HGdeviceptr dptr);

template <bool CheckSuccess> HGresult memAllocHost(void **pp, size_t bytesize);

template <bool CheckSuccess>
HGresult memHostAlloc(void **pp, size_t bytesize, unsigned int flags);

template <bool CheckSuccess>
HGresult memHostGetDevicePointer(HGdeviceptr *pdptr, void *p, unsigned int flags);

template <bool CheckSuccess> HGresult memFreeHost(void *p);

template <bool CheckSuccess>
HGresult launchKernel(HGfunction f, unsigned int gridDimX, unsigned int gridDimY,
                      unsigned int gridDimZ, unsigned int blockDimX,
                      unsigned int blockDimY, unsigned int blockDimZ,
                      unsigned int sharedMemBytes, HGstream hStream,
                      void **kernelParams, void **extra);

Device getDevice(uint64_t index);

} // namespace hggc

} // namespace proton

#endif // PROTON_DRIVER_GPU_HGGC_API_H_
