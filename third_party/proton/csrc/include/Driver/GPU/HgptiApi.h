#ifndef PROTON_DRIVER_HGPTI_API_H_
#define PROTON_DRIVER_HGPTI_API_H_

#include "Driver/Dispatch.h"
#include "hgpti.h"
#include "hgpti_pcsampling.h"

namespace proton {

namespace hgpti {

struct ExternLibHgpti : public ExternLibBase {
  using RetType = HGptiResult;
  static constexpr const char *name = "libhgpti.so";
  static constexpr const char *symbolName = "hgptiUnsubscribe";
  static constexpr const char *pathEnv = "TRITON_HGPTI_LIB_PATH";
  static constexpr RetType success = HGPTI_SUCCESS;
  static inline void *lib = nullptr;
};

template <bool CheckSuccess> HGptiResult getVersion(uint32_t *version);

template <bool CheckSuccess> HGptiResult getTimestamp(uint64_t *timestamp);

template <bool CheckSuccess>
HGptiResult getContextId(HGcontext context, uint32_t *pCtxId);

template <bool CheckSuccess>
HGptiResult activityRegisterCallbacks(
    HGpti_BuffersCallbackRequestFunc funcBufferRequested,
    HGpti_BuffersCallbackCompleteFunc funcBufferCompleted);

template <bool CheckSuccess>
HGptiResult subscribe(HGpti_SubscriberHandle *subscriber,
                      HGpti_CallbackFunc callback, void *userdata);

template <bool CheckSuccess>
HGptiResult enableDomain(uint32_t enable, HGpti_SubscriberHandle subscriber,
                         HGpti_CallbackDomain domain);

template <bool CheckSuccess>
HGptiResult enableCallback(uint32_t enable, HGpti_SubscriberHandle subscriber,
                           HGpti_CallbackDomain domain, HGpti_CallbackId cbid);

template <bool CheckSuccess>
HGptiResult activityEnableContext(HGcontext context, HGpti_ActivityKind kind);

template <bool CheckSuccess>
HGptiResult activityDisableContext(HGcontext context, HGpti_ActivityKind kind);

template <bool CheckSuccess>
HGptiResult activityEnable(HGpti_ActivityKind kind);

template <bool CheckSuccess>
HGptiResult activityDisable(HGpti_ActivityKind kind);

template <bool CheckSuccess> HGptiResult activityFlushAll(uint32_t flag);

template <bool CheckSuccess>
HGptiResult activityGetNextRecord(uint8_t *buffer, size_t validBufferSizeBytes,
                                  HGpti_Activity **record);

template <bool CheckSuccess>
HGptiResult activityPushExternalCorrelationId(HGpti_ExternalCorrelationKind kind,
                                              uint64_t id);

template <bool CheckSuccess>
HGptiResult activityPopExternalCorrelationId(HGpti_ExternalCorrelationKind kind,
                                             uint64_t *lastId);

template <bool CheckSuccess>
HGptiResult activitySetAttribute(HGpti_ActivityAttribute attr, size_t *valueSize,
                                 void *value);

template <bool CheckSuccess>
HGptiResult unsubscribe(HGpti_SubscriberHandle subscriber);

template <bool CheckSuccess> HGptiResult finalize();

template <bool CheckSuccess>
HGptiResult getGraphExecId(HGgraphExec graph, uint32_t *pId);

template <bool CheckSuccess>
HGptiResult getGraphId(HGgraph graph, uint32_t *pId);

template <bool CheckSuccess>
HGptiResult getGraphNodeId(HGgraphNode node, uint64_t *pId);

template <bool CheckSuccess>
HGptiResult getHgbinCrc(HGpti_GetHgbinCrcParams *pParams);

template <bool CheckSuccess>
HGptiResult
getSassToSourceCorrelation(HGpti_GetSassToSourceCorrelationParams *pParams);

template <bool CheckSuccess>
HGptiResult
pcSamplingGetNumStallReasons(HGpti_PCSamplingGetNumStallReasonsParams *pParams);

template <bool CheckSuccess>
HGptiResult
pcSamplingGetStallReasons(HGpti_PCSamplingGetStallReasonsParams *pParams);

template <bool CheckSuccess>
HGptiResult pcSamplingSetConfigurationAttribute(
    HGpti_PCSamplingConfigurationInfoParams *pParams);

template <bool CheckSuccess>
HGptiResult pcSamplingEnable(HGpti_PCSamplingEnableParams *pParams);

template <bool CheckSuccess>
HGptiResult pcSamplingDisable(HGpti_PCSamplingDisableParams *pParams);

template <bool CheckSuccess>
HGptiResult pcSamplingGetData(HGpti_PCSamplingGetDataParams *pParams);

template <bool CheckSuccess>
HGptiResult pcSamplingStart(HGpti_PCSamplingStartParams *pParams);

template <bool CheckSuccess>
HGptiResult pcSamplingStop(HGpti_PCSamplingStopParams *pParams);

} // namespace hgpti

} // namespace proton

#endif // PROTON_DRIVER_HGPTI_API_H_
