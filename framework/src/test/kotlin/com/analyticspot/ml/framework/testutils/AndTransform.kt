package com.analyticspot.ml.framework.testutils

import com.analyticspot.ml.framework.datatransform.StreamingDataTransform
import com.analyticspot.ml.framework.datatransform.TransformDescription
import com.analyticspot.ml.framework.description.ValueId
import com.analyticspot.ml.framework.description.ValueToken
import com.analyticspot.ml.framework.observation.Observation
import com.analyticspot.ml.framework.observation.SingleValueObservation

/**
 * This transform returns true if the values for all the source tokens passed to the contructor are true, false
 * otherwise.
 */
class AndTransform(val srcTokens: List<ValueToken<Boolean>>, val resultId: ValueId<Boolean>)
    : StreamingDataTransform() {
    override val description: TransformDescription = TransformDescription(listOf(ValueToken(resultId)))

    override fun transform(observation: Observation): Observation {
        val allTrue = srcTokens.map { observation.value(it) }.all { it }
        return SingleValueObservation.create(allTrue)
    }
}
