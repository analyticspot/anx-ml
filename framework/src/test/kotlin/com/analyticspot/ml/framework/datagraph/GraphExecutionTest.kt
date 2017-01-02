package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.dataset.IterableDataSet
import com.analyticspot.ml.framework.description.ValueId
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

            val trans = addTransform(src, AddConstantTransform(5, src.token(usedInput), resultId))
            result = trans
        }

        val srcObs = dg.buildSourceObservation("Hello not used value", 88)

        val transformF = dg.transform(srcObs, Executors.newSingleThreadExecutor())
        val resultObs = transformF.get()
        assertThat(resultObs.size).isEqualTo(1)
        assertThat(resultObs.value(dg.result.token(resultId))).isEqualTo(88 + 5)
    }

    @Test
    fun testSingleLearningTransformExecution() {
        val notUsedInput = ValueId.create<String>("notUsed")
        val usedInput = ValueId.create<Int>("used")
        val resultId = ValueId.create<Int>("finalResult")

        val dg = DataGraph.build {
            val src = setSource {
                valueIds += listOf(notUsedInput, usedInput)
            }

            val trans = addTransform(src, LearnMinTransform(src.token(usedInput), resultId))
            result = trans
        }

        val srcMatrix = listOf(
                listOf("Hello", 11),
                listOf("There", 22),
                listOf("Foo", 8),
                listOf("Bar", 4),
                listOf("Baz", 107)
        )
        val srcDataSet = IterableDataSet.create(srcMatrix)

        val transformF = dg.trainTransform(srcDataSet, Executors.newSingleThreadExecutor())
        val resultObs = transformF.get()
        val resultArrayOfObs = resultObs.toCollection(mutableListOf())
        assertThat(resultArrayOfObs.size).isEqualTo(srcMatrix.size)
        val outValues = resultObs.asSequence().map { it.value(dg.result.token(resultId)) }.toList()
        assertThat(outValues.min()).isEqualTo(4)
        assertThat(outValues.max()).isEqualTo(4)
        assertThat(outValues.size).isEqualTo(srcMatrix.size)
    }

}
