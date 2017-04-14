package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datatransform.DataTransform

/**
 * Created by oliver on 4/6/17.
 */
interface DelegatingTransform : DataTransform {
    val delegate: DataTransform
}
