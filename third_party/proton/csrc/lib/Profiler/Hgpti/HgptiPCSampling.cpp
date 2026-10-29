#include "Profiler/Hgpti/HgptiPCSampling.h"
#include "Data/Metric.h"
#include "Driver/GPU/HggcApi.h"
#include "Driver/GPU/HgptiApi.h"
#include "Utility/Atomic.h"
#include "Utility/Errors.h"
#include "Utility/Map.h"
#include "Utility/String.h"
#include <memory>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>

namespace proton {
namespace {

uint64_t getHgbinCrc(const char *hgbin, size_t size) {
  HGpti_GetHgbinCrcParams hgbinCrcParams{};
  hgbinCrcParams.size = HGpti_GetHgbinCrcParamsSize;
  hgbinCrcParams.hgbinSize = size;
  hgbinCrcParams.hgbin = hgbin;
  hgpti::getHgbinCrc<true>(&hgbinCrcParams);
  return hgbinCrcParams.hgbinCrc;
}

size_t getNumStallReasons(HGcontext context) {
  size_t numStallReasons = 0;
  HGpti_PCSamplingGetNumStallReasonsParams numStallReasonsParams{};
  numStallReasonsParams.size = HGpti_PCSamplingGetNumStallReasonsParamsSize;
  numStallReasonsParams.ctx = context;
  numStallReasonsParams.numStallReasons = &numStallReasons;
  hgpti::pcSamplingGetNumStallReasons<true>(&numStallReasonsParams);
  return numStallReasons;
}

std::optional<HgbinData::LineInfoValue>
getSassToSourceCorrelation(const char *functionName, uint64_t pcOffset,
                          const char *hgbin, size_t hgbinSize) {
  HGpti_GetSassToSourceCorrelationParams params{};
  params.size = HGpti_GetSassToSourceCorrelationParamsSize;
  params.hgbin = hgbin;
  params.functionName = functionName;
  params.hgbinSize = hgbinSize;
  params.pcOffset = pcOffset;
  auto status = hgpti::getSassToSourceCorrelation<false>(&params);
  std::optional<HgbinData::LineInfoValue> lineInfo;
  if (status == HGPTI_SUCCESS && params.fileName && params.fileName[0])
    lineInfo.emplace(params.lineNumber, functionName,
                     params.dirName ? params.dirName : "", params.fileName);
  std::free(params.fileName);
  std::free(params.dirName);
  return lineInfo;
}

std::pair<char **, uint32_t *>
getStallReasonNamesAndIndices(HGcontext context, size_t numStallReasons) {
  char **stallReasonNames =
      static_cast<char **>(std::calloc(numStallReasons, sizeof(char *)));
  for (size_t i = 0; i < numStallReasons; i++) {
    stallReasonNames[i] = static_cast<char *>(
        std::calloc(HGPTI_STALL_REASON_STRING_SIZE, sizeof(char)));
  }
  uint32_t *stallReasonIndices =
      static_cast<uint32_t *>(std::calloc(numStallReasons, sizeof(uint32_t)));

  HGpti_PCSamplingGetStallReasonsParams stallReasonsParams{};
  stallReasonsParams.size = HGpti_PCSamplingGetStallReasonsParamsSize;
  stallReasonsParams.ctx = context;
  stallReasonsParams.numStallReasons = numStallReasons;
  stallReasonsParams.stallReasonIndex = stallReasonIndices;
  stallReasonsParams.stallReasons = stallReasonNames;
  hgpti::pcSamplingGetStallReasons<true>(&stallReasonsParams);
  return std::make_pair(stallReasonNames, stallReasonIndices);
}

size_t matchStallReasonsToIndices(
    size_t numStallReasons, char **stallReasonNames, uint32_t *stallReasonIndices,
    std::map<size_t, size_t> &stallReasonIndexToMetricIndex,
    std::set<size_t> &notIssuedStallReasonIndices) {
  std::vector<bool> validIndex(numStallReasons, false);
  size_t numValidStalls = 0;
  for (size_t i = 0; i < numStallReasons; i++) {
    std::string_view hgptiStallName(stallReasonNames[i]);
    constexpr std::string_view prefix = "pu__pcsamp_warps_issue_";
    constexpr std::string_view suffix = "_not_issued";
    if (hgptiStallName.starts_with(prefix))
      hgptiStallName.remove_prefix(prefix.size());
    bool notIssued = hgptiStallName.ends_with(suffix);
    if (notIssued)
      hgptiStallName.remove_suffix(suffix.size());
    for (size_t j = 0; j < PCSamplingMetric::PCSamplingMetricKind::Count; j++) {
      auto metricName = PCSamplingMetric().getValueName(j);
      if (hgptiStallName == metricName) {
        if (notIssued)
          notIssuedStallReasonIndices.insert(stallReasonIndices[i]);
        stallReasonIndexToMetricIndex[stallReasonIndices[i]] = j;
        validIndex[i] = true;
        numValidStalls++;
        break;
      }
    }
  }
  int invalidIndex = -1;
  for (size_t i = 0; i < numStallReasons; i++) {
    if (invalidIndex == -1 && !validIndex[i]) {
      invalidIndex = i;
    } else if (invalidIndex != -1 && validIndex[i]) {
      std::swap(stallReasonIndices[invalidIndex], stallReasonIndices[i]);
      std::swap(stallReasonNames[invalidIndex], stallReasonNames[i]);
      validIndex[invalidIndex] = true;
      invalidIndex++;
    }
  }
  return numValidStalls;
}

HGpti_PCSamplingData allocPCSamplingData(size_t collectNumPCs,
                                        size_t numValidStallReasons) {
  HGpti_PCSamplingData pcSamplingData{};
  pcSamplingData.size = sizeof(HGpti_PCSamplingData);
  pcSamplingData.collectNumPcs = collectNumPCs;
  pcSamplingData.pPcData = static_cast<HGpti_PCSamplingPCData *>(
      std::calloc(collectNumPCs, sizeof(HGpti_PCSamplingPCData)));
  for (size_t i = 0; i < collectNumPCs; ++i) {
    pcSamplingData.pPcData[i].stallReason =
        static_cast<HGpti_PCSamplingStallReason *>(std::calloc(
            numValidStallReasons, sizeof(HGpti_PCSamplingStallReason)));
  }
  return pcSamplingData;
}

void enablePCSampling(HGcontext context) {
  HGpti_PCSamplingEnableParams params{};
  params.size = HGpti_PCSamplingEnableParamsSize;
  params.ctx = context;
  hgpti::pcSamplingEnable<true>(&params);
}

void disablePCSampling(HGcontext context) {
  HGpti_PCSamplingDisableParams params{};
  params.size = HGpti_PCSamplingDisableParamsSize;
  params.ctx = context;
  hgpti::pcSamplingDisable<true>(&params);
}

void startPCSampling(HGcontext context) {
  HGpti_PCSamplingStartParams params{};
  params.size = HGpti_PCSamplingStartParamsSize;
  params.ctx = context;
  hgpti::pcSamplingStart<true>(&params);
}

void stopPCSampling(HGcontext context) {
  HGpti_PCSamplingStopParams params{};
  params.size = HGpti_PCSamplingStopParamsSize;
  params.ctx = context;
  hgpti::pcSamplingStop<true>(&params);
}

void getPCSamplingData(HGcontext context,
                       HGpti_PCSamplingData *pcSamplingData) {
  HGpti_PCSamplingGetDataParams params{};
  params.size = HGpti_PCSamplingGetDataParamsSize;
  params.ctx = context;
  params.pcSamplingData = pcSamplingData;
  hgpti::pcSamplingGetData<true>(&params);
}

void setConfigurationAttribute(
    HGcontext context,
    std::vector<HGpti_PCSamplingConfigurationInfo> &configurationInfos) {
  HGpti_PCSamplingConfigurationInfoParams infoParams{};
  infoParams.size = HGpti_PCSamplingConfigurationInfoParamsSize;
  infoParams.ctx = context;
  infoParams.numAttributes = configurationInfos.size();
  infoParams.pPCSamplingConfigurationInfo = configurationInfos.data();
  hgpti::pcSamplingSetConfigurationAttribute<true>(&infoParams);
}

}

HGpti_PCSamplingConfigurationInfo
HgptiConfigureData::configureStallReasons() {
  numStallReasons = getNumStallReasons(context);
  std::tie(this->stallReasonNames, this->stallReasonIndices) =
      getStallReasonNamesAndIndices(context, numStallReasons);
  numValidStallReasons = matchStallReasonsToIndices(
      numStallReasons, stallReasonNames, stallReasonIndices,
      stallReasonIndexToMetricIndex, notIssuedStallReasonIndices);
  HGpti_PCSamplingConfigurationInfo stallReasonInfo{};
  stallReasonInfo.attributeType =
      HGPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_STALL_REASON;
  stallReasonInfo.attributeData.stallReasonData.stallReasonCount =
      numValidStallReasons;
  stallReasonInfo.attributeData.stallReasonData.pStallReasonIndex =
      stallReasonIndices;
  return stallReasonInfo;
}

HGpti_PCSamplingConfigurationInfo
HgptiConfigureData::configureSamplingPeriod() {
  HGpti_PCSamplingConfigurationInfo samplingPeriodInfo{};
  samplingPeriodInfo.attributeType =
      HGPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD;
  samplingPeriodInfo.attributeData.samplingPeriodData.samplingPeriod =
      DefaultFrequency;
  return samplingPeriodInfo;
}

HGpti_PCSamplingConfigurationInfo
HgptiConfigureData::configureSamplingBuffer() {
  HGpti_PCSamplingConfigurationInfo samplingBufferInfo{};
  samplingBufferInfo.attributeType =
      HGPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER;
  this->pcSamplingData =
      allocPCSamplingData(DataBufferPCCount, numValidStallReasons);
  samplingBufferInfo.attributeData.samplingDataBufferData.samplingDataBuffer =
      &this->pcSamplingData;
  return samplingBufferInfo;
}

HGpti_PCSamplingConfigurationInfo
HgptiConfigureData::configureScratchBuffer() {
  HGpti_PCSamplingConfigurationInfo scratchBufferInfo{};
  scratchBufferInfo.attributeType =
      HGPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE;
  scratchBufferInfo.attributeData.scratchBufferSizeData.scratchBufferSize =
      ScratchBufferSize;
  return scratchBufferInfo;
}

HGpti_PCSamplingConfigurationInfo
HgptiConfigureData::configureHardwareBufferSize() {
  HGpti_PCSamplingConfigurationInfo hardwareBufferInfo{};
  hardwareBufferInfo.attributeType =
      HGPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_HARDWARE_BUFFER_SIZE;
  hardwareBufferInfo.attributeData.hardwareBufferSizeData.hardwareBufferSize =
      HardwareBufferSize;
  return hardwareBufferInfo;
}

HGpti_PCSamplingConfigurationInfo
HgptiConfigureData::configureStartStopControl() {
  HGpti_PCSamplingConfigurationInfo startStopControlInfo{};
  startStopControlInfo.attributeType =
      HGPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL;
  startStopControlInfo.attributeData.enableStartStopControlData
      .enableStartStopControl = true;
  return startStopControlInfo;
}

HGpti_PCSamplingConfigurationInfo
HgptiConfigureData::configureCollectionMode() {
  HGpti_PCSamplingConfigurationInfo collectionModeInfo{};
  collectionModeInfo.attributeType =
      HGPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE;
  collectionModeInfo.attributeData.collectionModeData.collectionMode =
      HGPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS;
  return collectionModeInfo;
}

void HgptiConfigureData::initialize(HGcontext context) {
  this->context = context;
  hgpti::getContextId<true>(context, &contextId);
  configurationInfos.emplace_back(configureStallReasons());
  configurationInfos.emplace_back(configureSamplingPeriod());
  configurationInfos.emplace_back(configureHardwareBufferSize());
  configurationInfos.emplace_back(configureScratchBuffer());
  configurationInfos.emplace_back(configureSamplingBuffer());
  configurationInfos.emplace_back(configureStartStopControl());
  configurationInfos.emplace_back(configureCollectionMode());
  setConfigurationAttribute(context, configurationInfos);
}

HgptiConfigureData *HgptiPCSampling::getConfigureData(uint32_t contextId) {
  return &contextIdToConfigureData[contextId];
}

HgbinData *HgptiPCSampling::getHgbinData(uint64_t hgbinCrc) {
  if (!hgbinCrc)
    return nullptr;
  auto data = hgbinCrcToHgbinData.find(hgbinCrc);
  return data ? &data->get().first : nullptr;
}

void HgptiPCSampling::initialize(HGcontext context) {
  uint32_t contextId = 0;
  hgpti::getContextId<true>(context, &contextId);
  doubleCheckedLock(
      [&]() { return !contextInitialized.contains(contextId); }, contextMutex,
      [&]() {
        enablePCSampling(context);
        getConfigureData(contextId)->initialize(context);
        contextInitialized.insert(contextId);
      });
}

void HgptiPCSampling::start(HGcontext context) {
  uint32_t contextId = 0;
  hgpti::getContextId<true>(context, &contextId);
  doubleCheckedLock(
      [&]() -> bool { return !pcSamplingStarted; }, pcSamplingMutex,
      [&]() {
        initialize(context);
        hggc::ctxSynchronize<true>();
        startPCSampling(context);
        pcSamplingStarted = true;
      });
}

void HgptiPCSampling::processPCSamplingData(HgptiConfigureData *configureData,
                                          const DataToEntryMap &dataToEntry) {
  auto *pcSamplingData = &configureData->pcSamplingData;

  bool firstRound = true;
  while (pcSamplingData->totalNumPcs > 0 ||
         pcSamplingData->remainingNumPcs > 0 || firstRound) {
    for (size_t i = 0; i < pcSamplingData->totalNumPcs; ++i) {
      auto *pcData = pcSamplingData->pPcData + i;
      const HgbinData::LineInfoValue *lineInfo = nullptr;
      auto *hgbinData = getHgbinData(pcData->hgbinCrc);
      if (pcData->functionName && pcData->functionName[0] && hgbinData &&
          !hgbinData->hgbin.empty()) {
        auto key =
            HgbinData::LineInfoKey{pcData->functionIndex, pcData->pcOffset};
        auto it = hgbinData->lineInfo.find(key);
        if (it == hgbinData->lineInfo.end()) {
          auto source = getSassToSourceCorrelation(
              pcData->functionName, pcData->pcOffset, hgbinData->hgbin.data(),
              hgbinData->hgbin.size());
          if (source)
            it = hgbinData->lineInfo.try_emplace(key, std::move(*source)).first;
        }
        if (it != hgbinData->lineInfo.end())
          lineInfo = &it->second;
      }
      for (size_t j = 0; j < pcData->stallReasonCount; ++j) {
        auto *stallReason = &pcData->stallReason[j];
        if (!configureData->stallReasonIndexToMetricIndex.contains(
                stallReason->pcSamplingStallReasonIndex))
          throw makeOutOfRange("Invalid stall reason index");
        for (const auto &[data, baseEntry] : dataToEntry) {
          auto entry = baseEntry;
          if (lineInfo)
            entry = data->addOp(
                entry.phase, entry.id,
                {formatFileLineFunction(
                    lineInfo->dirName + "/" + lineInfo->fileName,
                    lineInfo->lineNumber, lineInfo->functionName)});
          auto metricKind = static_cast<PCSamplingMetric::PCSamplingMetricKind>(
              configureData->stallReasonIndexToMetricIndex
                  [stallReason->pcSamplingStallReasonIndex]);
          auto samples = stallReason->samples;
          auto stalledSamples =
              configureData->notIssuedStallReasonIndices.contains(
                  stallReason->pcSamplingStallReasonIndex)
                  ? 0
                  : samples;
          entry.upsertMetric(std::make_unique<PCSamplingMetric>(
              metricKind, samples, stalledSamples));
        }
      }
    }
    // HGPTI can leave the consumed batch in this reusable buffer.
    pcSamplingData->totalNumPcs = 0;
    if (pcSamplingData->remainingNumPcs > 0 || firstRound) {
      getPCSamplingData(configureData->context, pcSamplingData);
      firstRound = false;
    } else
      break;
  }
}

void HgptiPCSampling::stop(HGcontext context,
                           const DataToEntryMap &dataToEntry) {
  uint32_t contextId = 0;
  hgpti::getContextId<true>(context, &contextId);
  doubleCheckedLock(
      [&]() -> bool { return pcSamplingStarted; }, pcSamplingMutex,
      [&]() {
        hggc::ctxSynchronize<true>();
        auto *configureData = getConfigureData(contextId);
        stopPCSampling(context);
        pcSamplingStarted = false;
        processPCSamplingData(configureData, dataToEntry);
      });
}

void HgptiPCSampling::finalize(HGcontext context) {
  uint32_t contextId = 0;
  hgpti::getContextId<true>(context, &contextId);
  std::lock_guard<std::mutex> lock(contextMutex);
  if (!contextInitialized.contains(contextId))
    return;
  // HGPTI may still write to the buffers until disable has completed.
  disablePCSampling(context);
  contextIdToConfigureData.erase(contextId);
  contextInitialized.erase(contextId);
  // Loaded modules survive profiling sessions and remain needed for source correlation.
}

void HgptiPCSampling::loadModule(const char *hgbin, size_t hgbinSize) {
  if (!hgbin || !hgbinSize)
    return;
  auto hgbinCrc = getHgbinCrc(hgbin, hgbinSize);
  if (!hgbinCrc)
    return;
  hgbinCrcToHgbinData.upsert(hgbinCrc, [&](auto &module) {
    auto &[hgbinData, count] = module;
    if (count == 0) {
      hgbinData.hgbinCrc = hgbinCrc;
      hgbinData.hgbin.assign(hgbin, hgbinSize);
    }
    ++count;
  });
}

void HgptiPCSampling::unloadModule(const char *hgbin, size_t hgbinSize) {
  if (!hgbin || !hgbinSize)
    return;
  auto hgbinCrc = getHgbinCrc(hgbin, hgbinSize);
  auto module = hgbinCrcToHgbinData.find(hgbinCrc);
  if (!module)
    return;
  auto &count = module->get().second;
  if (count > 1)
    --count;
  else
    hgbinCrcToHgbinData.erase(hgbinCrc);
}

}
