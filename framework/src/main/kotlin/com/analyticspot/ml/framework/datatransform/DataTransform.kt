package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.serialization.Format
import com.analyticspot.ml.framework.serialization.StandardJsonFormat
import com.fasterxml.jackson.annotation.JsonIgnore
import java.util.concurrent.CompletableFuture

/**
 * A class for transformations that take one input DataSet and produce one output DataSet. These transforms do no learn
 * from training data; they simply make a transformation. See [LearningTransform] or similar for transformation that
 * need to be trained before they can be used.
 */
interface DataTransform {
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

    fun transform(dataSet: DataSet): CompletableFuture<DataSet>
}

