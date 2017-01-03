package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet
import java.util.concurrent.CompletableFuture

/**
 * A [DataTransform] that operates on a single input [DataSet]. As noted in [MultiTransform], most transforms should
 * implement [SingleDataTransform] and, if they need multiple inputs, use a [MergeTransform].
 */
interface SingleDataTransform : DataTransform {
    fun transform(dataSet: DataSet): CompletableFuture<DataSet>
}
