#include "Driver/GPU/HgptiApi.h"

namespace proton {

namespace hgpti {

DEFINE_DISPATCH(ExternLibHgpti, getVersion, hgptiGetVersion, uint32_t *)

DEFINE_DISPATCH(ExternLibHgpti, getTimestamp, hgptiGetTimestamp, uint64_t *)

DEFINE_DISPATCH(ExternLibHgpti, getContextId, hgptiGetContextId, HGcontext,
                uint32_t *)

DEFINE_DISPATCH(ExternLibHgpti, activityRegisterCallbacks,
                hgptiActivityRegisterCallbacks,
                HGpti_BuffersCallbackRequestFunc,
                HGpti_BuffersCallbackCompleteFunc)

DEFINE_DISPATCH(ExternLibHgpti, subscribe, hgptiSubscribe,
                HGpti_SubscriberHandle *, HGpti_CallbackFunc, void *)

DEFINE_DISPATCH(ExternLibHgpti, enableDomain, hgptiEnableDomain, uint32_t,
                HGpti_SubscriberHandle, HGpti_CallbackDomain)

DEFINE_DISPATCH(ExternLibHgpti, enableCallback, hgptiEnableCallback, uint32_t,
                HGpti_SubscriberHandle, HGpti_CallbackDomain, HGpti_CallbackId)

DEFINE_DISPATCH(ExternLibHgpti, activityEnable, hgptiActivityEnable,
                HGpti_ActivityKind)

DEFINE_DISPATCH(ExternLibHgpti, activityDisable, hgptiActivityDisable,
                HGpti_ActivityKind)

DEFINE_DISPATCH(ExternLibHgpti, activityEnableContext,
                hgptiActivityEnableContext, HGcontext, HGpti_ActivityKind)

DEFINE_DISPATCH(ExternLibHgpti, activityDisableContext,
                hgptiActivityDisableContext, HGcontext, HGpti_ActivityKind)

DEFINE_DISPATCH(ExternLibHgpti, activityFlushAll, hgptiActivityFlushAll, uint32_t)

DEFINE_DISPATCH(ExternLibHgpti, activityGetNextRecord,
                hgptiActivityGetNextRecord, uint8_t *, size_t, HGpti_Activity **)

DEFINE_DISPATCH(ExternLibHgpti, activityPushExternalCorrelationId,
                hgptiActivityPushExternalCorrelationId,
                HGpti_ExternalCorrelationKind, uint64_t)

DEFINE_DISPATCH(ExternLibHgpti, activityPopExternalCorrelationId,
                hgptiActivityPopExternalCorrelationId,
                HGpti_ExternalCorrelationKind, uint64_t *)

DEFINE_DISPATCH(ExternLibHgpti, activitySetAttribute, hgptiActivitySetAttribute,
                HGpti_ActivityAttribute, size_t *, void *)

DEFINE_DISPATCH(ExternLibHgpti, unsubscribe, hgptiUnsubscribe,
                HGpti_SubscriberHandle)

DEFINE_DISPATCH(ExternLibHgpti, finalize, hgptiFinalize)

DEFINE_DISPATCH(ExternLibHgpti, getGraphExecId, hgptiGetGraphExecId, HGgraphExec,
                uint32_t *)

DEFINE_DISPATCH(ExternLibHgpti, getGraphId, hgptiGetGraphId, HGgraph, uint32_t *)

DEFINE_DISPATCH(ExternLibHgpti, getGraphNodeId, hgptiGetGraphNodeId, HGgraphNode,
                uint64_t *)

DEFINE_DISPATCH(ExternLibHgpti, getHgbinCrc, hgptiGetHgbinCrc,
                HGpti_GetHgbinCrcParams *)

DEFINE_DISPATCH(ExternLibHgpti, getSassToSourceCorrelation,
                hgptiGetSassToSourceCorrelation,
                HGpti_GetSassToSourceCorrelationParams *)

DEFINE_DISPATCH(ExternLibHgpti, pcSamplingGetNumStallReasons,
                hgptiPCSamplingGetNumStallReasons,
                HGpti_PCSamplingGetNumStallReasonsParams *)

DEFINE_DISPATCH(ExternLibHgpti, pcSamplingGetStallReasons,
                hgptiPCSamplingGetStallReasons,
                HGpti_PCSamplingGetStallReasonsParams *)

DEFINE_DISPATCH(ExternLibHgpti, pcSamplingSetConfigurationAttribute,
                hgptiPCSamplingSetConfigurationAttribute,
                HGpti_PCSamplingConfigurationInfoParams *)

DEFINE_DISPATCH(ExternLibHgpti, pcSamplingEnable, hgptiPCSamplingEnable,
                HGpti_PCSamplingEnableParams *)

DEFINE_DISPATCH(ExternLibHgpti, pcSamplingDisable, hgptiPCSamplingDisable,
                HGpti_PCSamplingDisableParams *)

DEFINE_DISPATCH(ExternLibHgpti, pcSamplingGetData, hgptiPCSamplingGetData,
                HGpti_PCSamplingGetDataParams *)

DEFINE_DISPATCH(ExternLibHgpti, pcSamplingStart, hgptiPCSamplingStart,
                HGpti_PCSamplingStartParams *)

DEFINE_DISPATCH(ExternLibHgpti, pcSamplingStop, hgptiPCSamplingStop,
                HGpti_PCSamplingStopParams *)

} // namespace hgpti

} // namespace proton
