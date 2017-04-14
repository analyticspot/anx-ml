package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datatransform.DataTransform
import com.fasterxml.jackson.annotation.JsonIgnore
import java.io.OutputStream

/**
 * Interface that must be implemented by classes that use the [MultiFileMixedFormat] for serialization.
 */
interface MultiFileMixedTransform : DataTransform {
    override val formatClass: Class<out Format>
        @JsonIgnore
        get() = MultiFileMixedFormat::class.java
    /**
     * The transform should write binary data to the given output stream when this is called.
     */
    fun serializeBinaryData(output: OutputStream)
}
