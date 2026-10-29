#ifndef PROTON_PROFILER_HGPTI_PROFILER_H_
#define PROTON_PROFILER_HGPTI_PROFILER_H_

#include "Profiler/GPUProfiler.h"

namespace proton {

class HgptiProfiler final : public GPUProfiler<HgptiProfiler> {
public:
  ~HgptiProfiler() override;

private:
  friend class Singleton<HgptiProfiler>;

  HgptiProfiler();

  struct HgptiProfilerPimpl;

  void doSetMode(const std::vector<std::string> &modeAndOptions) override;
};

}

#endif
