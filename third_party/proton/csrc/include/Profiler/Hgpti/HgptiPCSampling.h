#ifndef PROTON_PROFILER_HGPTI_PC_SAMPLING_H_
#define PROTON_PROFILER_HGPTI_PC_SAMPLING_H_

#include "Driver/GPU/HggcApi.h"
#include "Driver/GPU/HgptiApi.h"
#include "Profiler/Hgpti/HgptiProfiler.h"
#include "Utility/Map.h"
#include "Utility/Set.h"
#include "Utility/Singleton.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace proton {

struct HgbinData {
  uint64_t hgbinCrc{};
  std::string hgbin;

  struct LineInfoKey {
    uint32_t functionIndex;
    uint64_t pcOffset;

    bool operator<(const LineInfoKey &other) const {
      return functionIndex < other.functionIndex ||
             (functionIndex == other.functionIndex && pcOffset < other.pcOffset);
    }
  };

  struct LineInfoValue {
    uint32_t lineNumber{};
    const std::string functionName{};
    const std::string dirName{};
    const std::string fileName{};

    LineInfoValue() = default;

    LineInfoValue(uint32_t lineNumber, const std::string &functionName,
                  const std::string &dirName, const std::string &fileName)
        : lineNumber(lineNumber), functionName(functionName), dirName(dirName),
          fileName(fileName) {}
  };

  std::map<LineInfoKey, LineInfoValue> lineInfo;
};

struct HgptiConfigureData {
  HgptiConfigureData() = default;

  ~HgptiConfigureData() {
    if (stallReasonNames) {
      for (size_t i = 0; i < numStallReasons; i++) {
        if (stallReasonNames[i])
          std::free(stallReasonNames[i]);
      }
      std::free(stallReasonNames);
    }
    if (stallReasonIndices)
      std::free(stallReasonIndices);
    if (pcSamplingData.pPcData) {
      for (size_t i = 0; i < DataBufferPCCount; ++i) {
        std::free(pcSamplingData.pPcData[i].stallReason);
      }
      std::free(pcSamplingData.pPcData);
    }
  }

  void initialize(HGcontext context);

  HGpti_PCSamplingConfigurationInfo configureStallReasons();
  HGpti_PCSamplingConfigurationInfo configureSamplingPeriod();
  HGpti_PCSamplingConfigurationInfo configureSamplingBuffer();
  HGpti_PCSamplingConfigurationInfo configureScratchBuffer();
  HGpti_PCSamplingConfigurationInfo configureHardwareBufferSize();
  HGpti_PCSamplingConfigurationInfo configureStartStopControl();
  HGpti_PCSamplingConfigurationInfo configureCollectionMode();

  static constexpr size_t HardwareBufferSize = 128 * 1024 * 1024;
  static constexpr size_t ScratchBufferSize = 16 * 1024 * 1024;
  static constexpr size_t DataBufferPCCount = 1024;
  static constexpr uint32_t DefaultFrequency = 10;

  HGcontext context{};
  uint32_t contextId{};
  uint32_t numStallReasons{};
  uint32_t numValidStallReasons{};
  char **stallReasonNames{};
  uint32_t *stallReasonIndices{};
  std::map<size_t, size_t> stallReasonIndexToMetricIndex{};
  std::set<size_t> notIssuedStallReasonIndices{};
  HGpti_PCSamplingData pcSamplingData{};

  // HGPTI retains these configuration pointers until sampling is disabled.
  std::vector<HGpti_PCSamplingConfigurationInfo> configurationInfos;
};

class HgptiPCSampling : public Singleton<HgptiPCSampling> {
public:
  HgptiPCSampling() = default;
  virtual ~HgptiPCSampling() = default;

  void initialize(HGcontext context);

  void start(HGcontext context);

  void stop(HGcontext context, const DataToEntryMap &dataToEntry);

  void finalize(HGcontext context);

  void loadModule(const char *hgbin, size_t hgbinSize);

  void unloadModule(const char *hgbin, size_t hgbinSize);

private:
  HgptiConfigureData *getConfigureData(uint32_t contextId);

  HgbinData *getHgbinData(uint64_t hgbinCrc);

  void processPCSamplingData(HgptiConfigureData *configureData,
                             const DataToEntryMap &dataToEntry);

  ThreadSafeMap<uint32_t, HgptiConfigureData> contextIdToConfigureData;

  ThreadSafeMap<uint64_t, std::pair<HgbinData, size_t>> hgbinCrcToHgbinData;
  ThreadSafeSet<uint32_t> contextInitialized;

  std::atomic<bool> pcSamplingStarted{false};
  std::mutex pcSamplingMutex{};
  std::mutex contextMutex{};
};

}

#endif
