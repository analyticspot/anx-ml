package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.description.ValueIdGroup
import com.analyticspot.ml.framework.description.ValueToken

/**
 * Describes the outputs produced by a transform. The main purpose here is to allow each [DataTransform] to specify
 * the [ValueToken] types that will be most efficent for the data structure they produce.
 */
data class TransformDescription(val tokens: List<ValueToken<*>>, val tokenGroups: List<ValueIdGroup<*>> = listOf()) {
}
