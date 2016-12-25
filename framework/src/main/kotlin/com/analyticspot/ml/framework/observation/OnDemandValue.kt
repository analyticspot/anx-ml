package com.analyticspot.ml.framework.observation

import java.util.concurrent.CompletableFuture

/**
 * A value in an [Observation] that is only computed on demand. This is typically used for features that are expensive
 * to compute so we only want to compute them if the value is necessary to classify the current [Observation].
 */
interface OnDemandValue<DataT> {
    /**
     * Start the computation of the feature and returns a future that will complete with the value.
     */
    fun demand(): CompletableFuture<DataT>

}
