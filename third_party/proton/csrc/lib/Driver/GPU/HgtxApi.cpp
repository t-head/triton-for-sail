#include "Driver/GPU/HgtxApi.h"
#include "Driver/GPU/HgptiApi.h"

#include <cstdlib>

namespace proton {

namespace {

struct RangePushAParams {
  const char *message;
};

} // namespace

namespace hgtx {

void enable() {
  const std::string hgptiLibPath =
      Dispatch<hgpti::ExternLibHgpti>::getLibPath();
  if (!hgptiLibPath.empty()) {
    setenv("HGTX_INJECTION64_PATH", hgptiLibPath.c_str(), 1);
    setenv("NVTX_INJECTION64_PATH", hgptiLibPath.c_str(), 1);
  }
}

void disable() {
  unsetenv("HGTX_INJECTION64_PATH");
  unsetenv("NVTX_INJECTION64_PATH");
}

std::string getMessageFromRangePushA(const void *params) {
  if (const auto *p = static_cast<const RangePushAParams *>(params))
    return std::string(p->message ? p->message : "");
  return "";
}

} // namespace hgtx

} // namespace proton
