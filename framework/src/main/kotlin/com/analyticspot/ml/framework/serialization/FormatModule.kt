package com.analyticspot.ml.framework.serialization

/**
 *
 */
interface FormatModule<T : FormatModuleData> {
    fun getFactory(formatData: T, tag: String): TransformFactory
}
