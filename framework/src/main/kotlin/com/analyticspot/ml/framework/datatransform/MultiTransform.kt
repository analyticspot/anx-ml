package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.serialization.Format
import com.analyticspot.ml.framework.serialization.StandardJsonFormat
import com.fasterxml.jackson.annotation.JsonIgnore

/**
 * A [MultiTransform] is like a [DataTransform] but it takes more than one [DataSet] as input. It is recommended to
 * minimize use of [MultiTransform] as they are more complicated to write and much more complex to correctly
 * deserialize (due to issues with mapping [ValueId] to [ValueToken]. For the most part users should be able to write
 * regular [DataTransform] nodes and then use [MergeTransform] when inputs from multiple [DataSet]s are required.
 */
interface MultiTransform {
    /**
     * Describes the outputs produced by this transformation.
     */
    @get:JsonIgnore
    val description: TransformDescription

    /**
     * The format to which this node serializes. By default this is [StandardJsonFormat].
     */
    val formatClass: Class<out Format<*>>
        @JsonIgnore
        get() = StandardJsonFormat::class.java

    /**
     * Like [DataTransform.transform] but takes a list of [DataSet] as input.
     */
    fun transform(dataSets: List<DataSet>): DataSet
}
