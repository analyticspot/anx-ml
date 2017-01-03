package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet
import java.util.concurrent.CompletableFuture

/**
 * A [MultiTransform] is like a [SingleDataTransform] but it takes more than one [DataSet] as input. It is recommended
 * to minimize use of [MultiTransform] as they are more complicated to write and much more complex to correctly
 * deserialize (due to issues with mapping [ValueId] to [ValueToken]. For the most part users should be able to write
 * regular [DataTransform] nodes and then use [MergeTransform] when inputs from multiple [DataSet]s are required.
 */
interface MultiTransform : DataTransform {
    companion object {
        /**
         * Used with `@JacksonInject` to indicate where you want the list of source
         * [DataSet] to be injected if using [StandardJsonFormat].
         */
        const val JSON_SOURCE_INJECTION_ID = "MultiTransformSources"
    }
    /**
     * Like [SingleDataTransform.transform] but takes a list of [DataSet] as input.
     */
    fun transform(dataSets: List<DataSet>): CompletableFuture<DataSet>
}
