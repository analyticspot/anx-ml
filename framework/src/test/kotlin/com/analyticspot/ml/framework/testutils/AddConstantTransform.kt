package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.datatransform.StreamingDataTransform
import com.analyticspot.ml.framework.datatransform.TransformDescription
import com.analyticspot.ml.framework.description.ValueId
import com.analyticspot.ml.framework.description.ValueToken
import com.analyticspot.ml.framework.observation.Observation
import com.analyticspot.ml.framework.observation.SingleValueObservation
import com.fasterxml.jackson.annotation.JacksonInject

class AddConstantTransform(val toAdd: Int, val srcToken: ValueToken<Int>, val resultId: ValueId<Int>)
    : StreamingDataTransform() {
    private val resultToken = ValueToken(resultId)
    override val description = TransformDescription(listOf(resultToken))

    override fun transform(observation: Observation): Observation {
        val srcVal: Int = observation.value(srcToken)
        return SingleValueObservation.create(srcVal + toAdd)
    }

    class DeserBuilder(@JacksonInject private val sources: List<GraphNode>) {

    }
}
