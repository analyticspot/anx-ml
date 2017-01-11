package com.analyticspot.ml.framework.description

import com.analyticspot.ml.framework.datatransform.StreamingDataTransform
import com.analyticspot.ml.framework.description.TransformDescription
import com.analyticspot.ml.framework.observation.Observation
import com.analyticspot.ml.framework.observation.SingleValueObservation

/**
 * Silly transform that just returns a list of all the tokens in a group (with the prefix removed). Helpful just for
 * testing.
 */
class TokenNamesTrans(val srcGroup: ValueTokenGroup<*>, val resultId: ValueId<String>) : StreamingDataTransform() {
    override val description: TransformDescription
        get() = TransformDescription(listOf(ValueToken(resultId)))

    override fun transform(observation: Observation): Observation {
        val prefix = srcGroup.name
        val tokenNames = mutableListOf<String>()
        srcGroup.tokens().forEach {
            tokenNames.add(it.name.removePrefix(prefix).removePrefix(ValueId.GROUP_SEPARATOR))
        }
        return SingleValueObservation(tokenNames.joinToString(" "), String::class.java)
    }
}
