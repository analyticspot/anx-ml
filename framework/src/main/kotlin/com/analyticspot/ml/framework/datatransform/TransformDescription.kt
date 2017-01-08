package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.description.ValueToken
import com.analyticspot.ml.framework.description.ValueTokenGroup

/**
 * Describes the outputs produced by an execution of the node. The main purpose here is to allow each [DataTransform] to
 * specify the [ValueToken] types that will be most efficient for the data structure they produce.
 */
data class TransformDescription(val tokens: List<ValueToken<*>>,
        val tokenGroups: List<ValueTokenGroup<*>> = listOf()) {
}
