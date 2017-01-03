package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.dataset.IterableDataSet
import com.analyticspot.ml.framework.description.ValueId
import com.analyticspot.ml.framework.observation.SingleValueObservation
import org.assertj.core.api.Assertions.assertThat
import org.slf4j.LoggerFactory
import org.testng.annotations.BeforeClass
import org.testng.annotations.Test
import java.util.concurrent.Executors

class GraphExecutionTest {
    companion object {
        private val log = LoggerFactory.getLogger(GraphExecutionTest::class.java)
    }

    @BeforeClass
    fun setup() {
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            log.error("Thread {} threw an error:", thread.id, throwable)
        }
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

    @Test
    fun testMergeTransfomExecution() {
        val srcValId = ValueId.create<Int>("source")

        var mergeDs: GraphNode? = null

        val resultValIds = listOf(
                ValueId.create<Int>("t1"),
                ValueId.create<Int>("t2"),
                ValueId.create<Int>("t3")
        )

        val dg = DataGraph.build {
            val src = setSource {
                valueIds += srcValId
            }

            // 3 parallel transforms that add 1, 2 and 3 respectively
            val t1 = addTransform(src, AddConstantTransform(1, src.token(srcValId), resultValIds[0]))
            val t2 = addTransform(src, AddConstantTransform(2, src.token(srcValId), resultValIds[1]))
            val t3 = addTransform(src, AddConstantTransform(3, src.token(srcValId), resultValIds[2]))

            mergeDs = merge(t1, t2, t3)

            result = mergeDs ?: throw AssertionError("Should be non-null here")
        }

        val nnMergeDs = mergeDs ?: throw AssertionError("data set should not be null")
        assertThat(nnMergeDs.tokens).hasSize(3)
        assertThat(nnMergeDs.tokens.map { it.name }.toSet()).isEqualTo(resultValIds.map { it.name }.toSet())

        val resultF = dg.transform(SingleValueObservation.create(0), Executors.newFixedThreadPool(3))
        val resultObs = resultF.get()
        assertThat(resultObs.size).isEqualTo(3)
        val resultTokens = resultValIds.map { dg.result.token(it) }
        assertThat(resultObs.value(resultTokens[0])).isEqualTo(1)
        assertThat(resultObs.value(resultTokens[1])).isEqualTo(2)
        assertThat(resultObs.value(resultTokens[2])).isEqualTo(3)
    }

}
