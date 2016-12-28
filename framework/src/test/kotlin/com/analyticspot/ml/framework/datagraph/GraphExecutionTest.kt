package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.datatransform.StreamingDataTransform
import com.analyticspot.ml.framework.datatransform.TransformDescription
import com.analyticspot.ml.framework.description.ValueId
import com.analyticspot.ml.framework.description.ValueToken
import com.analyticspot.ml.framework.observation.Observation
import com.analyticspot.ml.framework.observation.SingleValueObservation
import org.assertj.core.api.Assertions.assertThat
import org.slf4j.LoggerFactory
import org.testng.annotations.Test
import java.util.concurrent.Executors

class GraphExecutionTest {
    companion object {
        private val log = LoggerFactory.getLogger(GraphExecutionTest::class.java)
    }

    @Test
    fun testSingleTransformExecution() {
        val notUsedInput = ValueId.create<String>("notUsed")
        val usedInput = ValueId.create<Int>("used")
        val resultId = ValueId.create<Int>("finalResult")

        val dg = DataGraph.build {
            val src = setSource {
                valueIds += listOf(notUsedInput, usedInput)
            }

            val trans = addTransform(src, AddFiveTransform(src.token(usedInput), resultId))
            result = trans
        }

        val srcObs = dg.buildTransformSource("Hello not used value", 88)

        val transformF = dg.transform(srcObs, Executors.newSingleThreadExecutor())
        val resultObs = transformF.get()
        assertThat(resultObs.size).isEqualTo(1)
        assertThat(resultObs.value(dg.result.token(resultId))).isEqualTo(88 + 5)
    }

    class AddFiveTransform(private val srcToken: ValueToken<Int>, resultId: ValueId<Int>) : StreamingDataTransform() {
        private val resultToken = ValueToken(resultId)
        override val description = TransformDescription(listOf(resultToken))

        override fun transform(observation: Observation): Observation {
            val srcVal: Int = observation.value(srcToken)
            return SingleValueObservation.create(srcVal + 5)
        }
    }
}
