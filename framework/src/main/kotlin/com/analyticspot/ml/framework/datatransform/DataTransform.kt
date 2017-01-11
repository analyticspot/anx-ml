package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.description.TransformDescription
import com.analyticspot.ml.framework.serialization.Format
import com.analyticspot.ml.framework.serialization.StandardJsonFormat
import com.fasterxml.jackson.annotation.JsonIgnore

/**
 * Base interface for all tranformations.
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
}

