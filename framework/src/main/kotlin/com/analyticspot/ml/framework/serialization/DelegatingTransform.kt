package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datatransform.DataTransform

/**
 * Interface for transforms which use the [DelegatingFormat] for serialization and deserialization.
 */
interface DelegatingTransform : DataTransform {
    val delegate: DataTransform
}
