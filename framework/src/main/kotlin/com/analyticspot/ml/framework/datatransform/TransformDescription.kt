package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.description.ValueIdGroup
import com.analyticspot.ml.framework.description.ValueToken

/**
 * Describes the outputs produced by a transform.
 */
data class TransformDescription(val tokens: List<ValueToken<*>>, val tokenGroups: List<ValueIdGroup<*>> = listOf()) {
}
