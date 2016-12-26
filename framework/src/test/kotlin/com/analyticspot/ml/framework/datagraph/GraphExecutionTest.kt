package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.datatransform.DataTransform
import com.analyticspot.ml.framework.datatransform.TransformDescription
import com.analyticspot.ml.framework.description.IndexValueToken
import com.analyticspot.ml.framework.description.ValueToken
import com.analyticspot.ml.framework.observation.Observation
import com.analyticspot.ml.framework.observation.SingleValueObservation
import org.assertj.core.api.Assertions.assertThat
import org.testng.annotations.Test
import java.util.concurrent.Executors

class GraphExecutionTest {
    @Test
    fun testSingleTransformExecution() {
        val dg = DataGraph.build {
            val src = setSource {
                tokens += listOf(
                        IndexValueToken.create<String>(0, "notUsed"),
                        IndexValueToken.create<Integer>(1, "used"))
            }

            val trans = addTransform(src, AddFiveTransform(src.token("used"), "finalResult"))
            result = trans
        }

        val srcObs = dg.buildTransformSource("Hello not used value", 88)

        val transformF = dg.transform(srcObs, Executors.newSingleThreadExecutor())
        val resultObs = transformF.get()
        assertThat(resultObs.size).isEqualTo(1)
        assertThat(resultObs.value(dg.result.token<Integer>("finalResult"))).isEqualTo(88 + 5)
    }

    class AddFiveTransform(private val srcToken: ValueToken<Integer>, resultName: String) : DataTransform {
        private val resultToken = ValueToken.create<Integer>(resultName)
        override val description = TransformDescription(listOf(resultToken))

        override fun transform(observation: Observation): Observation {
            val srcVal: Integer = observation.value(srcToken)
            return SingleValueObservation.create(Integer.valueOf(srcVal.toInt() + 5))
        }
    }
}
