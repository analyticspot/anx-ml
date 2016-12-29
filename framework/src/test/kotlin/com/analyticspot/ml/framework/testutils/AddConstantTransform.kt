package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.datatransform.StreamingDataTransform
import com.analyticspot.ml.framework.datatransform.TransformDescription
import com.analyticspot.ml.framework.description.ValueId
import com.analyticspot.ml.framework.description.ValueToken
import com.analyticspot.ml.framework.observation.Observation
import com.analyticspot.ml.framework.observation.SingleValueObservation

class AddConstantTransform(private val toAdd: Int, private val srcToken: ValueToken<Int>, resultId: ValueId<Int>) : StreamingDataTransform() {
    private val resultToken = ValueToken(resultId)
    override val description = TransformDescription(listOf(resultToken))

    override fun transform(observation: Observation): Observation {
        val srcVal: Int = observation.value(srcToken)
        return SingleValueObservation.create(srcVal + toAdd)
    }
}
