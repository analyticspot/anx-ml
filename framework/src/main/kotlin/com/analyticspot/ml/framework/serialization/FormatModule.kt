package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datatransform.DataTransform
import java.io.OutputStream

/**
 *
 */
interface FormatModule<T : FormatData> {
    fun serialize(transform: DataTransform, output: OutputStream)

    fun formatData(transform: DataTransform): FormatData

    fun getFactory(tag: String?): TransformFactory<T>

    /**
     * Allows the user to register a factory that will be used to deserialize [DataTransform] instances serialized with
     * the given tag instead of the default deserialization for the module.
     */
    fun registerFactory(tag: String, factory: TransformFactory<T>)
}
