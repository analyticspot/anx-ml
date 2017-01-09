package com.analyticspot.ml.framework.testutils

import com.analyticspot.ml.framework.datatransform.StreamingDataTransform
import com.analyticspot.ml.framework.datatransform.TransformDescription
import com.analyticspot.ml.framework.description.ValueId
import com.analyticspot.ml.framework.description.ValueToken
import com.analyticspot.ml.framework.observation.Observation
import com.analyticspot.ml.framework.observation.SingleValueObservation

/**
 * Takes a String input and converts it to lowercase.
 */
class LowerCaseTransform(private val srcToken: ValueToken<String>,
        private val resultId: ValueId<String>) : StreamingDataTransform() {
    override val description: TransformDescription = TransformDescription(listOf(ValueToken(resultId)))

    override fun transform(observation: Observation): Observation {
        val toLowercase = observation.value(srcToken)
        return SingleValueObservation.create(toLowercase.toLowerCase())
    }
}
