#include "Runtime/HggcRuntime.h"

#include "Driver/GPU/HggcApi.h"
#include <algorithm>
#include <cstdint>

namespace proton {

void HggcRuntime::launchKernel(void *kernel, unsigned int gridDimX,
                             unsigned int gridDimY, unsigned int gridDimZ,
                             unsigned int blockDimX, unsigned int blockDimY,
                             unsigned int blockDimZ,
                             unsigned int sharedMemBytes, void *stream,
                             void **kernelParams, void **extra) {
  hggc::launchKernel<true>(reinterpret_cast<HGfunction>(kernel), gridDimX, gridDimY,
                         gridDimZ, blockDimX, blockDimY, blockDimZ,
                         sharedMemBytes, reinterpret_cast<HGstream>(stream),
                         kernelParams, extra);
}

void HggcRuntime::memset(void *devicePtr, uint32_t value, size_t size,
                       void *stream) {
  hggc::memsetD32Async<true>(reinterpret_cast<HGdeviceptr>(devicePtr), value,
                          size / sizeof(uint32_t),
                          reinterpret_cast<HGstream>(stream));
}

void HggcRuntime::allocateHostBuffer(uint8_t **buffer, size_t size, bool mapped) {
  if (mapped) {
    hggc::memHostAlloc<true>(reinterpret_cast<void **>(buffer), size,
                           HG_MEMHOSTALLOC_DEVICEMAP | HG_MEMHOSTALLOC_PORTABLE);
  } else {
    hggc::memAllocHost<true>(reinterpret_cast<void **>(buffer), size);
  }
}

void HggcRuntime::getHostDevicePointer(uint8_t *hostPtr, uint8_t **devicePtr) {
  HGdeviceptr devicePtrV;
  hggc::memHostGetDevicePointer<true>(&devicePtrV, hostPtr, 0);
  *devicePtr = reinterpret_cast<uint8_t *>(devicePtrV);
}

void HggcRuntime::freeHostBuffer(uint8_t *buffer) {
  hggc::memFreeHost<true>(buffer);
}

void HggcRuntime::allocateDeviceBuffer(uint8_t **buffer, size_t size) {
  HGdeviceptr devicePtr;
  hggc::memAlloc<true>(&devicePtr, size);
  *buffer = reinterpret_cast<uint8_t *>(devicePtr);
}

void HggcRuntime::freeDeviceBuffer(uint8_t *buffer) {
  HGdeviceptr devicePtr = reinterpret_cast<HGdeviceptr>(buffer);
  hggc::memFree<true>(devicePtr);
}

void HggcRuntime::copyDeviceToHostAsync(void *dst, const void *src, size_t size,
                                     void *stream) {
  hggc::memcpyDToHAsync<true>(dst, reinterpret_cast<HGdeviceptr>(src), size,
                           reinterpret_cast<HGstream>(stream));
}

void *HggcRuntime::getDevice() {
  HGdevice device;
  hggc::ctxGetDevice<true>(&device);
  return reinterpret_cast<void *>(static_cast<uintptr_t>(device));
}

void *HggcRuntime::getPriorityStream() {
  HGstream stream;
  int lowestPriority, highestPriority;
  hggc::ctxGetStreamPriorityRange<true>(&lowestPriority, &highestPriority);
  hggc::streamCreateWithPriority<true>(&stream, HG_STREAM_NON_BLOCKING,
                                     highestPriority);
  return reinterpret_cast<void *>(stream);
}

void HggcRuntime::synchronizeStream(void *stream) {
  hggc::streamSynchronize<true>(reinterpret_cast<HGstream>(stream));
}

void HggcRuntime::destroyStream(void *stream) {
  hggc::streamDestroy<true>(reinterpret_cast<HGstream>(stream));
}

void HggcRuntime::synchronizeDevice() {
  HGcontext context = nullptr;
  hggc::ctxGetCurrent<false>(&context);
  if (context) {
    hggc::ctxSynchronize<true>();
  }
}

void HggcRuntime::processHostBuffer(
    uint8_t *hostBuffer, size_t hostBufferSize, uint8_t *deviceBuffer,
    size_t deviceBufferSize, void *stream,
    std::function<void(uint8_t *, size_t)> callback) {
  int64_t chunkSize = std::min(hostBufferSize, deviceBufferSize);
  int64_t sizeLeftOnDevice = deviceBufferSize;
  while (chunkSize > 0) {
    hggc::memcpyDToHAsync<true>(reinterpret_cast<void *>(hostBuffer),
                             reinterpret_cast<HGdeviceptr>(deviceBuffer),
                             chunkSize, reinterpret_cast<HGstream>(stream));
    hggc::streamSynchronize<true>(reinterpret_cast<HGstream>(stream));
    callback(hostBuffer, chunkSize);
    hostBuffer += chunkSize;
    deviceBuffer += chunkSize;
    sizeLeftOnDevice -= chunkSize;
    chunkSize =
        std::min(static_cast<int64_t>(hostBufferSize), sizeLeftOnDevice);
  }
}

} // namespace proton
