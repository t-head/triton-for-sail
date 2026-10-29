#include "Profiler/Hgpti/HgptiProfiler.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Device.h"
#include "Driver/GPU/HggcApi.h"
#include "Driver/GPU/HgptiApi.h"
#include "Driver/GPU/HgtxApi.h"
#include "Profiler/Graph.h"
#include "Profiler/Hgpti/HgptiPCSampling.h"
#include "Runtime/HggcRuntime.h"
#include "Utility/Env.h"
#include "Utility/Errors.h"
#include "Utility/Map.h"
#include "Utility/String.h"
#include "Utility/Vector.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace proton {
template <>
thread_local GPUProfiler<HgptiProfiler>::ThreadState
    GPUProfiler<HgptiProfiler>::threadState(HgptiProfiler::instance());

namespace {

std::unique_ptr<Metric>
convertKernelActivityToMetric(HGpti_Activity *activity,
                              bool isMetricKernel = false) {
  std::unique_ptr<Metric> metric;
  auto *kernel = reinterpret_cast<HGpti_ActivityKernel5 *>(activity);
  if (kernel->start < kernel->end) {
    metric = std::make_unique<KernelMetric>(
        static_cast<uint64_t>(kernel->start), static_cast<uint64_t>(kernel->end),
        1, static_cast<uint64_t>(kernel->deviceId),
        static_cast<uint64_t>(DeviceType::PPU),
        static_cast<uint64_t>(kernel->streamId),
        static_cast<uint64_t>(isMetricKernel));
  }
  return metric;
}

struct ActivityCompletion {
  uint64_t correlationId{};
  bool countsAsTask{};
};

ActivityCompletion
processActivityKernel(CorrIdToExternIdMap &corrIdToExternId,
                      ExternIdToStateMap &externIdToState,
                      std::map<uint64_t, std::reference_wrapper<ExternIdState>>
                          &externIdToStateCache,
                      DataPhases &dataPhases, HGpti_Activity *activity) {
  auto *kernel = reinterpret_cast<HGpti_ActivityKernel5 *>(activity);
  auto correlationId = kernel->correlationId;
  size_t externId = 0;
  if (!corrIdToExternId.withRead(
          correlationId, [&externId](size_t value) { externId = value; })) {
    corrIdToExternId.erase(correlationId);
    return {correlationId, false};
  }
  if (kernel->graphId == 0) {
    bool isMissingName = false;
    DataToEntryMap dataToEntry;
    externIdToState.withRead(externId, [&](const ExternIdState &state) {
      isMissingName = state.isMissingName;
      dataToEntry = state.dataToEntry;
    });
    if (!isMissingName) {
      for (auto &[data, entry] : dataToEntry) {
        if (auto kernelMetric = convertKernelActivityToMetric(activity)) {
          entry.upsertMetric(std::move(kernelMetric));
          detail::updateDataPhases(dataPhases, data, entry.phase);
        }
      }
    } else {
      for (auto &[data, entry] : dataToEntry) {
        if (auto kernelMetric = convertKernelActivityToMetric(activity)) {
          auto childEntry =
              data->addOp(entry.phase, entry.id, {Context(kernel->name)});
          childEntry.upsertMetric(std::move(kernelMetric));
          detail::updateDataPhases(dataPhases, data, entry.phase);
        }
      }
    }
    externIdToState.erase(externId);
    corrIdToExternId.erase(correlationId);
    return {correlationId, true};
  } else {
    auto iter = externIdToStateCache.find(externId);
    ExternIdState *state = nullptr;
    if (iter != externIdToStateCache.end()) {
      state = &iter->second.get();
    } else {
      auto ref = externIdToState.find(externId);

      externIdToStateCache.emplace(externId, ref.value());
      state = &ref.value().get();
    }
    auto &externState = *state;

    auto *nodeIdToState = externState.nodeIdToState;
    if (nodeIdToState) {
      const GraphState::NodeState &nodeState =
          nodeIdToState->at(kernel->graphNodeId);
      if (nodeState.status.isMissingName()) {
        throw makeLogicError("Kernel name is missing for a graph node.");
      }
      const bool isMetricKernel = nodeState.status.isMetricNode();
      for (auto &[data, entry] : externState.dataToGraphEntry) {
        auto targetEntryIdIter = nodeState.dataToEntryId.find(data);
        if (targetEntryIdIter != nodeState.dataToEntryId.end()) {
          auto targetEntryId = targetEntryIdIter->second;
          if (auto kernelMetric =
                  convertKernelActivityToMetric(activity, isMetricKernel)) {
            entry.upsertLinkedMetric(std::move(kernelMetric), targetEntryId);
            detail::updateDataPhases(dataPhases, data, entry.phase);
          }
        }
      }
    } else {
      for (auto &[data, entry] : externState.dataToEntry) {
        if (auto kernelMetric = convertKernelActivityToMetric(activity)) {
          auto childEntry =
              data->addOp(entry.phase, entry.id, {Context(kernel->name)});
          childEntry.upsertMetric(std::move(kernelMetric));
          detail::updateDataPhases(dataPhases, data, childEntry.phase);
        }
      }
    }

    if (externState.numNodes > 0) {
      externState.numNodes--;
    }

    if (externState.numNodes == 0) {
      externIdToState.erase(externId);
      corrIdToExternId.erase(correlationId);
    }
    return {correlationId, true};
  }
}

ActivityCompletion
processActivity(CorrIdToExternIdMap &corrIdToExternId,
                ExternIdToStateMap &externIdToState,
                std::map<uint64_t, std::reference_wrapper<ExternIdState>>
                    &externIdToStateCache,
                DataPhases &dataPhases, HGpti_Activity *activity) {
  switch (activity->kind) {
  case HGPTI_ACTIVITY_KIND_KERNEL:
  case HGPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
    return processActivityKernel(corrIdToExternId, externIdToState,
                                 externIdToStateCache, dataPhases, activity);
  }
  default:
    break;
  }
  return {};
}

constexpr std::array<HGpti_CallbackId, 11> kGraphCallbacks = {
    HGPTI_HGGC_DRIVER_CBID_hgGraphLaunch,
    HGPTI_HGGC_DRIVER_CBID_hgGraphLaunch_ptsz,
    HGPTI_HGGC_DRIVER_CBID_hgStreamBeginCapture,
    HGPTI_HGGC_DRIVER_CBID_hgStreamBeginCapture_ptsz,
    HGPTI_HGGC_DRIVER_CBID_hgStreamEndCapture,
    HGPTI_HGGC_DRIVER_CBID_hgStreamEndCapture_ptsz,
    HGPTI_HGGC_DRIVER_CBID_hgStreamBeginCapture_v2,
    HGPTI_HGGC_DRIVER_CBID_hgStreamBeginCapture_v2_ptsz,
    HGPTI_HGGC_DRIVER_CBID_hgStreamBeginCaptureToGraph,
    HGPTI_HGGC_DRIVER_CBID_hgStreamBeginCaptureToGraph_ptsz,
    HGPTI_HGGC_DRIVER_CBID_hgStreamEndCapture};

#define PROTON_KERNEL_CALLBACK_LIST(X)                                        \
  X(HGPTI_HGGC_DRIVER_CBID_hgLaunch)                                          \
  X(HGPTI_HGGC_DRIVER_CBID_hgLaunchGrid)                                      \
  X(HGPTI_HGGC_DRIVER_CBID_hgLaunchGridAsync)                                 \
  X(HGPTI_HGGC_DRIVER_CBID_hgLaunchKernel)                                    \
  X(HGPTI_HGGC_DRIVER_CBID_hgLaunchKernel_ptsz)                               \
  X(HGPTI_HGGC_DRIVER_CBID_hgLaunchKernelEx)                                  \
  X(HGPTI_HGGC_DRIVER_CBID_hgLaunchKernelEx_ptsz)                             \
  X(HGPTI_HGGC_DRIVER_CBID_hgLaunchCooperativeKernel)                         \
  X(HGPTI_HGGC_DRIVER_CBID_hgLaunchCooperativeKernel_ptsz)                    \
  X(HGPTI_HGGC_DRIVER_CBID_hgLaunchCooperativeKernelMultiDevice)

#define PROTON_KERNEL_CB_AS_ID(cbId) cbId,
constexpr std::array<HGpti_CallbackId, 10> kKernelCallbacks = {
    PROTON_KERNEL_CALLBACK_LIST(PROTON_KERNEL_CB_AS_ID)};
#undef PROTON_KERNEL_CB_AS_ID

constexpr std::array<HGpti_CallbackId, 2> kGraphResourceCallbacks = {
    HGPTI_CBID_RESOURCE_GRAPHNODE_CREATED,
    HGPTI_CBID_RESOURCE_GRAPHNODE_CLONED};

constexpr std::array<HGpti_CallbackId, 4> kResourceCallbacks = {
    HGPTI_CBID_RESOURCE_MODULE_LOADED,
    HGPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING,
    HGPTI_CBID_RESOURCE_CONTEXT_CREATED,
    HGPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING,
};

constexpr std::array<HGpti_CallbackId, 2> kHgtxCallbacks = {
    HGPTI_CBID_HGTX_hgtxRangePushA,
    HGPTI_CBID_HGTX_hgtxRangePop,
};

void setLaunchCallbacks(HGpti_SubscriberHandle subscriber, bool enable) {
  for (auto cbId : kKernelCallbacks) {
    hgpti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,
                                HGPTI_CB_DOMAIN_DRIVER_API, cbId);
  }
}

void setGraphCallbacks(HGpti_SubscriberHandle subscriber, bool enable) {
  for (auto cbId : kGraphCallbacks) {
    hgpti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,
                                HGPTI_CB_DOMAIN_DRIVER_API, cbId);
  }
  for (auto cbId : kGraphResourceCallbacks) {
    hgpti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,
                                HGPTI_CB_DOMAIN_RESOURCE, cbId);
  }
}

void setResourceCallbacks(HGpti_SubscriberHandle subscriber, bool enable) {
  for (auto cbId : kResourceCallbacks) {
    hgpti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,
                                HGPTI_CB_DOMAIN_RESOURCE, cbId);
  }
}

void setHgtxCallbacks(HGpti_SubscriberHandle subscriber, bool enable) {
  for (auto cbId : kHgtxCallbacks) {
    hgpti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,
                                HGPTI_CB_DOMAIN_HGTX, cbId);
  }
}

bool isKernel(HGpti_CallbackId cbId) {
  switch (cbId) {
#define PROTON_KERNEL_CB_AS_CASE(cbId)                                        \
  case cbId:                                                                  \
    return true;
    PROTON_KERNEL_CALLBACK_LIST(PROTON_KERNEL_CB_AS_CASE)
#undef PROTON_KERNEL_CB_AS_CASE
  default:
    return false;
  }
}

bool isGraphLaunch(HGpti_CallbackId cbId) {
  return cbId == HGPTI_HGGC_DRIVER_CBID_hgGraphLaunch ||
         cbId == HGPTI_HGGC_DRIVER_CBID_hgGraphLaunch_ptsz;
}

bool isLaunch(HGpti_CallbackId cbId) {
  return isKernel(cbId) || isGraphLaunch(cbId);
}

bool isStreamCaptureBegin(HGpti_CallbackId cbId) {
  return cbId == HGPTI_HGGC_DRIVER_CBID_hgStreamBeginCapture ||
         cbId == HGPTI_HGGC_DRIVER_CBID_hgStreamBeginCapture_ptsz ||
         cbId == HGPTI_HGGC_DRIVER_CBID_hgStreamBeginCapture_v2 ||
         cbId == HGPTI_HGGC_DRIVER_CBID_hgStreamBeginCapture_v2_ptsz ||
         cbId == HGPTI_HGGC_DRIVER_CBID_hgStreamBeginCaptureToGraph ||
         cbId == HGPTI_HGGC_DRIVER_CBID_hgStreamBeginCaptureToGraph_ptsz;
}

bool isStreamCaptureEnd(HGpti_CallbackId cbId) {
  return cbId == HGPTI_HGGC_DRIVER_CBID_hgStreamEndCapture ||
         cbId == HGPTI_HGGC_DRIVER_CBID_hgStreamEndCapture_ptsz;
}

#undef PROTON_KERNEL_CALLBACK_LIST

}

struct HgptiProfiler::HgptiProfilerPimpl
    : public GPUProfiler<HgptiProfiler>::GPUProfilerPimplInterface {
  HgptiProfilerPimpl(HgptiProfiler &profiler)
      : GPUProfiler<HgptiProfiler>::GPUProfilerPimplInterface(profiler) {
    auto runtime = &HggcRuntime::instance();
    profiler.metricBuffer = std::make_unique<MetricBuffer>(
        getIntEnv("TRITON_PROFILE_METRIC_BUFFER_SIZE", 64 * 1024 * 1024),
        runtime, true);
    profiler.pendingGraphPool =
        std::make_unique<PendingGraphPool>(profiler.metricBuffer.get());
  }
  ~HgptiProfilerPimpl() override = default;

  void doStart() override;
  void doFlush() override;
  void doStop() override;

  static void allocBuffer(uint8_t **buffer, size_t *bufferSize,
                          size_t *maxNumRecords);
  static void completeBuffer(HGcontext context, uint32_t streamId,
                             uint8_t *buffer, size_t size, size_t validSize);
  static void callbackFn(void *userData, HGpti_CallbackDomain domain,
                         HGpti_CallbackId cbId, const void *cbData);

  static constexpr size_t AlignSize = 8;
  static constexpr size_t AttributeSize = sizeof(size_t);
  static constexpr const char *CaptureTag = "<captured_at>";

  HGpti_SubscriberHandle subscriber{};
  HgptiPCSampling pcSampling;

  ThreadSafeMap<uint64_t, GraphState> graphStates;

private:
  void handleGraphResourceCallbacks(HgptiProfiler &profiler,
                                    HGpti_CallbackId cbId,
                                    HGpti_GraphData *graphData);
  void handleResourceCallbacks(HgptiProfiler &profiler, HGpti_CallbackId cbId,
                               const void *cbData);
  void handleHgtxCallbacks(HGpti_CallbackId cbId, const void *cbData);

  void handleStreamCaptureCallbacks(HGpti_CallbackId cbId,
                                    const HGpti_CallbackData *callbackData);
  void handleApiEnterLaunchCallbacks(HgptiProfiler &profiler,
                                     HGpti_CallbackId cbId,
                                     const HGpti_CallbackData *callbackData);
  void handleApiExitLaunchCallbacks(HgptiProfiler &profiler,
                                    HGpti_CallbackId cbId,
                                    const HGpti_CallbackData *callbackData);
  void handleApiCallbacks(HgptiProfiler &profiler, HGpti_CallbackId cbId,
                          const void *cbData);
};

void HgptiProfiler::HgptiProfilerPimpl::allocBuffer(uint8_t **buffer,
                                                  size_t *bufferSize,
                                                  size_t *maxNumRecords) {
  const auto envBufferSize =
      getIntEnv("TRITON_PROFILE_BUFFER_SIZE", 64 * 1024 * 1024);
  *buffer = static_cast<uint8_t *>(aligned_alloc(AlignSize, envBufferSize));
  if (*buffer == nullptr) {
    throw makeRuntimeError("aligned_alloc failed");
  }
  *bufferSize = envBufferSize;
  *maxNumRecords = 0;
}

void HgptiProfiler::HgptiProfilerPimpl::completeBuffer(
    HGcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size,
    size_t validSize) {
  HgptiProfiler &profiler = threadState.profiler;
  uint64_t numCompletedTasks = 0;
  uint64_t maxCorrelationId = 0;
  DataPhases dataPhases;
  HGptiResult status;
  HGpti_Activity *activity = nullptr;
  std::map<uint64_t, std::reference_wrapper<ExternIdState>>
      externIdToStateCache;
  do {
    status = hgpti::activityGetNextRecord<false>(buffer, validSize, &activity);
    if (status == HGPTI_SUCCESS) {
      auto completion =
          processActivity(profiler.correlation.corrIdToExternId,
                          profiler.correlation.externIdToState,
                          externIdToStateCache, dataPhases, activity);
      numCompletedTasks += completion.countsAsTask;
      maxCorrelationId = std::max(maxCorrelationId, completion.correlationId);
    } else if (status == HGPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    } else {
      throw makeRuntimeError("hgpti::activityGetNextRecord failed");
    }
  } while (true);

  std::free(buffer);

  profiler.correlation.complete(numCompletedTasks, maxCorrelationId);
  profiler.flushDataPhases(dataPhases, profiler.pendingGraphPool.get());
}

void HgptiProfiler::HgptiProfilerPimpl::handleGraphResourceCallbacks(
    HgptiProfiler &profiler, HGpti_CallbackId cbId, HGpti_GraphData *graphData) {
  uint32_t graphId = 0;
  uint32_t graphExecId = 0;
  if (graphData->graph)
    hgpti::getGraphId<true>(graphData->graph, &graphId);
  if (graphData->graphExec)
    hgpti::getGraphExecId<true>(graphData->graphExec, &graphExecId);
  if (cbId == HGPTI_CBID_RESOURCE_GRAPHNODE_CREATED ||
      cbId == HGPTI_CBID_RESOURCE_GRAPHNODE_CLONED) {
    if (graphData->nodeType != HG_GRAPH_NODE_TYPE_KERNEL) {
      return;
    }
    uint64_t nodeId = 0;
    hgpti::getGraphNodeId<true>(graphData->node, &nodeId);
    if (cbId == HGPTI_CBID_RESOURCE_GRAPHNODE_CREATED) {
      if (!profiler.isOpInProgress()) {
        return;
      }
      auto &graphState = graphStates[graphId];
      std::optional<GraphState::MetricNodeState> metricNodeState;
      if (threadState.isMetricKernelLaunching) {
        auto metricKernelLaunchInfo =
            threadState.metricKernelLaunchInfoQueue.front();
        threadState.metricKernelLaunchInfoQueue.pop_front();
        metricNodeState.emplace(GraphState::MetricNodeState{
            metricKernelLaunchInfo.seqId, metricKernelLaunchInfo.metricId,
            metricKernelLaunchInfo.numWords});
      }
      graphState.recordNode(nodeId, threadState.scopeStack.back().name,
                            std::move(metricNodeState), profiler.dataSet,
                            threadState.isApiExternOp);
    } else {
      uint32_t originalGraphId = 0;
      uint64_t originalNodeId = 0;
      hgpti::getGraphId<true>(graphData->originalGraph, &originalGraphId);
      hgpti::getGraphNodeId<true>(graphData->originalNode, &originalNodeId);
      auto &originalGraphState = graphStates[originalGraphId];
      auto &graphState = graphStates[graphId];
      graphState.nodeIdToState[nodeId] =
          originalGraphState.nodeIdToState[originalNodeId];
      auto &nodeState = graphState.nodeIdToState[nodeId];
      nodeState.nodeId = nodeId;
      for (const auto &dataToEntryId : nodeState.dataToEntryId)
        graphState.capturedData.insert(dataToEntryId.first);
      auto originalMetricNodeIt =
          originalGraphState.metricNodeIdToState.find(originalNodeId);
      if (originalMetricNodeIt !=
          originalGraphState.metricNodeIdToState.end()) {
        auto metricNodeState = originalMetricNodeIt->second;
        graphState.metricNodeIdToState.insert_or_assign(nodeId, metricNodeState);
        graphState.metricSeqIdToNodeId.insert_or_assign(metricNodeState.seqId,
                                                       nodeId);
        graphState.numMetricWords += metricNodeState.numWords;
      }
    }
  }
}

void HgptiProfiler::HgptiProfilerPimpl::handleResourceCallbacks(
    HgptiProfiler &profiler, HGpti_CallbackId cbId, const void *cbData) {
  auto *resourceData =
      static_cast<HGpti_ResourceData *>(const_cast<void *>(cbData));
  if (cbId == HGPTI_CBID_RESOURCE_MODULE_LOADED) {
    auto *moduleResource = static_cast<HGpti_ModuleResourceData *>(
        resourceData->resourceDescriptor);
    if (profiler.pcSamplingEnabled)
      pcSampling.loadModule(moduleResource->pCubin, moduleResource->cubinSize);
  } else if (cbId == HGPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING) {
    auto *moduleResource = static_cast<HGpti_ModuleResourceData *>(
        resourceData->resourceDescriptor);
    if (profiler.pcSamplingEnabled)
      pcSampling.unloadModule(moduleResource->pCubin, moduleResource->cubinSize);
  } else if (cbId == HGPTI_CBID_RESOURCE_CONTEXT_CREATED) {
    if (profiler.pcSamplingEnabled)
      pcSampling.initialize(resourceData->context);
  } else if (cbId == HGPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING) {
    if (profiler.pcSamplingEnabled)
      pcSampling.finalize(resourceData->context);
  } else {
    auto *graphData =
        static_cast<HGpti_GraphData *>(resourceData->resourceDescriptor);
    handleGraphResourceCallbacks(profiler, cbId, graphData);
  }
}

void HgptiProfiler::HgptiProfilerPimpl::handleHgtxCallbacks(
    HGpti_CallbackId cbId, const void *cbData) {
  auto *hgtxData = static_cast<const HGpti_HgtxData *>(cbData);
  if (cbId == HGPTI_CBID_HGTX_hgtxRangePushA) {
    auto message = hgtx::getMessageFromRangePushA(hgtxData->functionParams);
    threadState.enterScope(message);
  } else if (cbId == HGPTI_CBID_HGTX_hgtxRangePop) {
    threadState.exitScope();
  }
}

void HgptiProfiler::HgptiProfilerPimpl::handleStreamCaptureCallbacks(
    HGpti_CallbackId cbId, const HGpti_CallbackData *callbackData) {
  if (callbackData->callbackSite != HGPTI_API_ENTER)
    return;

  if (isStreamCaptureBegin(cbId)) {
    threadState.isStreamCapturing = true;
    profiler.metricBuffer->reserve();
  } else if (isStreamCaptureEnd(cbId)) {
    threadState.isStreamCapturing = false;
  }
}

void HgptiProfiler::HgptiProfilerPimpl::handleApiEnterLaunchCallbacks(
    HgptiProfiler &profiler, HGpti_CallbackId cbId,
    const HGpti_CallbackData *callbackData) {
  size_t numNodes = 1;
  if (isGraphLaunch(cbId)) {
    threadState.enterOp(Scope(""));
  } else {
    const auto symbolName = callbackData->context && callbackData->symbolName
                                ? std::string(callbackData->symbolName)
                                : "";
    threadState.enterOp(Scope(std::move(symbolName)));
  }

  auto &dataToEntry = threadState.dataToEntry;
  if (threadState.isStreamCapturing)
    return;

  const auto &scope = threadState.scopeStack.back();
  if (isGraphLaunch(cbId)) {
    auto graphExec =
        static_cast<const hgGraphLaunch_params *>(callbackData->functionParams)
            ->hGraph;
    uint32_t graphExecId = 0;
    hgpti::getGraphExecId<true>(graphExec, &graphExecId);
    numNodes = detail::prepareGraphLaunch(
        graphStates, graphExecId, scope.scopeId, dataToEntry,
        profiler.correlation.externIdToState, profiler.pendingGraphPool.get(),
        callbackData->context != nullptr);
  }

  if (dataToEntry.empty())
    return;

  profiler.correlation.correlate(callbackData->correlationId, scope.scopeId,
                                  numNodes, scope.name.empty(), dataToEntry);
  if (profiler.pcSamplingEnabled)
    pcSampling.start(callbackData->context);

  numNodes = numNodes == std::numeric_limits<size_t>::max() ? 1 : numNodes;
  profiler.correlation.submit(numNodes, callbackData->correlationId);
}

void HgptiProfiler::HgptiProfilerPimpl::handleApiExitLaunchCallbacks(
    HgptiProfiler &profiler, HGpti_CallbackId cbId,
    const HGpti_CallbackData *callbackData) {
  auto &dataToEntry = threadState.dataToEntry;
  if (profiler.pcSamplingEnabled) {
    pcSampling.stop(callbackData->context, dataToEntry);
  }
  threadState.exitOp();
}

void HgptiProfiler::HgptiProfilerPimpl::handleApiCallbacks(
    HgptiProfiler &profiler, HGpti_CallbackId cbId, const void *cbData) {
  const HGpti_CallbackData *callbackData =
      static_cast<const HGpti_CallbackData *>(cbData);
  handleStreamCaptureCallbacks(cbId, callbackData);
  if (isLaunch(cbId)) {
    if (callbackData->callbackSite == HGPTI_API_ENTER) {
      handleApiEnterLaunchCallbacks(profiler, cbId, callbackData);
    } else if (callbackData->callbackSite == HGPTI_API_EXIT) {
      handleApiExitLaunchCallbacks(profiler, cbId, callbackData);
    }
  }
}

void HgptiProfiler::HgptiProfilerPimpl::callbackFn(
    void *userData, HGpti_CallbackDomain domain, HGpti_CallbackId cbId,
    const void *cbData) {
  HgptiProfiler &profiler = threadState.profiler;
  auto *pImpl = dynamic_cast<HgptiProfilerPimpl *>(profiler.pImpl.get());
  if (domain == HGPTI_CB_DOMAIN_RESOURCE) {
    pImpl->handleResourceCallbacks(profiler, cbId, cbData);
  } else if (domain == HGPTI_CB_DOMAIN_HGTX) {
    pImpl->handleHgtxCallbacks(cbId, cbData);
  } else {
    pImpl->handleApiCallbacks(profiler, cbId, cbData);
  }
}

void HgptiProfiler::HgptiProfilerPimpl::doStart() {
  hgpti::subscribe<true>(&subscriber, callbackFn, nullptr);
  if (profiler.pcSamplingEnabled) {
    setResourceCallbacks(subscriber, true);
    HGcontext context = nullptr;
    hggc::ctxGetCurrent<true>(&context);
    if (context)
      pcSampling.initialize(context);
    hgpti::activityEnable<true>(HGPTI_ACTIVITY_KIND_KERNEL);
  } else {
    hgpti::activityEnable<true>(HGPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  }
  hgpti::activityRegisterCallbacks<true>(allocBuffer, completeBuffer);
  setGraphCallbacks(subscriber, true);
  setLaunchCallbacks(subscriber, true);
  if (getBoolEnv("TRITON_ENABLE_NVTX", true)) {
    hgtx::enable();
    setHgtxCallbacks(subscriber, true);
  }

  if (!profiler.timestampOffsetNs) {
    profiler.timestampOffsetNs =
        detail::computeTimestampOffsetNs(hgpti::getTimestamp<true>);
  }
}

void HgptiProfiler::HgptiProfilerPimpl::doFlush() {
  HGcontext hgContext = nullptr;
  hggc::ctxGetCurrent<false>(&hgContext);
  if (hgContext) {
    hggc::ctxSynchronize<true>();
  }
  profiler.correlation.flush(100, 10,
                             []() { hgpti::activityFlushAll<true>(0); });

  // Flush incomplete activities so they cannot leak into the next session.
  hgpti::activityFlushAll<true>(HGPTI_ACTIVITY_FLAG_FLUSH_FORCED);

  profiler.pendingGraphPool->flushAll();
}

void HgptiProfiler::HgptiProfilerPimpl::doStop() {
  if (profiler.pcSamplingEnabled) {
    profiler.pcSamplingEnabled = false;
    HGcontext hgContext = nullptr;
    hggc::ctxGetCurrent<false>(&hgContext);
    if (hgContext)
      pcSampling.finalize(hgContext);
    setResourceCallbacks(subscriber, false);
    hgpti::activityDisable<true>(HGPTI_ACTIVITY_KIND_KERNEL);
  } else {
    hgpti::activityDisable<true>(HGPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  }
  profiler.periodicFlushingEnabled = false;
  profiler.periodicFlushingFormat.clear();

  // HGPTI resets correlation IDs when unsubscribing.
  profiler.correlation.clear();
  setGraphCallbacks(subscriber, false);
  setLaunchCallbacks(subscriber, false);
  hgtx::disable();
  setHgtxCallbacks(subscriber, false);
  hgpti::unsubscribe<true>(subscriber);
  hgpti::finalize<true>();
}

HgptiProfiler::HgptiProfiler() {
  pImpl = std::make_unique<HgptiProfilerPimpl>(*this);
}

HgptiProfiler::~HgptiProfiler() = default;

void HgptiProfiler::doSetMode(const std::vector<std::string> &modeAndOptions) {
  if (getBoolEnv("TRITON_ENABLE_HW_TRACE", false))
    throw makeRuntimeError(
        "HgptiProfiler: TRITON_ENABLE_HW_TRACE is unsupported on PPU");
  auto mode = modeAndOptions[0];
  if (proton::toLower(mode) == "pcsampling") {
    pcSamplingEnabled = true;
  } else if (proton::toLower(mode) == "periodic_flushing") {
    detail::setPeriodicFlushingMode(periodicFlushingEnabled,
                                   periodicFlushingFormat, modeAndOptions,
                                   "HgptiProfiler");
  } else if (!mode.empty()) {
    throw makeInvalidArgument("HgptiProfiler: unsupported mode: " + mode);
  }
}

}
