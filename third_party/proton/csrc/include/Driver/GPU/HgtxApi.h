#ifndef PROTON_DRIVER_HGTX_API_H_
#define PROTON_DRIVER_HGTX_API_H_

#include <string>

namespace proton {

namespace hgtx {

void enable();

void disable();

std::string getMessageFromRangePushA(const void *params);

} // namespace hgtx

} // namespace proton

#endif // PROTON_DRIVER_HGTX_API_H_
