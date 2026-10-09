#include "Driver/GPU/NvtxApi.h"
#include "Driver/GPU/CuptiApi.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <unistd.h>

namespace proton {

namespace {

// Declare nvtx function params without including the nvtx header
struct RangePushAParams {
  const char *message;
};

} // namespace

namespace nvtx {

namespace {

bool isPPU() {
  // Check if ppu-smi exists in PATH, consistent with Python-side detection
  const char *path = std::getenv("PATH");
  if (!path)
    return false;
  std::string pathStr(path);
  std::string::size_type start = 0;
  while (start < pathStr.size()) {
    auto end = pathStr.find(':', start);
    if (end == std::string::npos)
      end = pathStr.size();
    std::string dir = pathStr.substr(start, end - start) + "/ppu-smi";
    if (access(dir.c_str(), X_OK) == 0)
      return true;
    start = end + 1;
  }
  return false;
}

} // namespace

void enable() {
  if (isPPU()) {
    const char *ppuSdk = std::getenv("PPU_SDK");
    std::string sdkPath = ppuSdk ? ppuSdk : "/usr/local/PPU_SDK";
    std::string injectionPath = sdkPath + "/asight/lib/libhggc_injection.so";
    setenv("NVTX_INJECTION64_PATH", injectionPath.c_str(), 1);
  } else {
    const std::string cuptiLibPath =
        Dispatch<cupti::ExternLibCupti>::getLibPath();
    if (!cuptiLibPath.empty()) {
      setenv("NVTX_INJECTION64_PATH", cuptiLibPath.c_str(), 1);
    }
  }
}

void disable() { unsetenv("NVTX_INJECTION64_PATH"); }

std::string getMessageFromRangePushA(const void *params) {
  if (const auto *p = static_cast<const RangePushAParams *>(params))
    return std::string(p->message ? p->message : "");
  return "";
}

} // namespace nvtx

} // namespace proton
