package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.datatransform.ColumnSubsetTransform
import com.analyticspot.ml.framework.datatransform.SingleDataTransform
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.description.TransformDescription
import com.analyticspot.ml.framework.testutils.Graph1
import com.analyticspot.ml.framework.testutils.InvertBoolean
import com.analyticspot.ml.framework.testutils.TrueIfSeenTransform
import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.slf4j.LoggerFactory
import org.testng.annotations.BeforeClass
import org.testng.annotations.Test
import java.util.concurrent.CompletableFuture
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
        val notUsedInput = ColumnId.create<String>("notUsed")
        val usedInput = ColumnId.create<Int>("used")
        val toAdd = 5

        val dg = DataGraph.build {
            val src = setSource {
                columnIds += listOf(notUsedInput, usedInput)
            }

            val trans = addTransform(src, AddConstantTransform(toAdd, src.transformDescription))
            result = trans
        }

        val srcData = dg.createSource("Hello not used value", 88)

        val transformF = dg.transform(srcData, Executors.newSingleThreadExecutor())
        val resultData = transformF.get()
        assertThat(resultData.numRows).isEqualTo(1)
        assertThat(resultData.columnIds).hasSize(2)
        assertThat(resultData.value(0, usedInput)).isEqualTo(srcData.value(0, usedInput)!! + toAdd)
        assertThat(resultData.value(0, notUsedInput)).isEqualTo(srcData.value(0, notUsedInput))
    }

     @Test
     fun testSingleLearningTransformExecution() {
         val notUsedInput = ColumnId.create<String>("notUsed")
         val usedInput = ColumnId.create<Int>("used")
         val resultId = ColumnId.create<Int>("finalResult")

         val dg = DataGraph.build {
             val src = setSource {
                 columnIds += listOf(notUsedInput, usedInput)
             }

             val trans = addTransform(src, LearnMinTransform(usedInput, resultId))
             result = trans
         }

         val srcMatrix = listOf(
                 listOf("Hello", 11),
                 listOf("There", 22),
                 listOf("Foo", 8),
                 listOf("Bar", 4),
                 listOf("Baz", 107)
         )
         val srcDataSet = dg.createSource(srcMatrix)

         val transformF = dg.trainTransform(srcDataSet, Executors.newSingleThreadExecutor())
         val resultData = transformF.get()
         assertThat(resultData.numRows).isEqualTo(srcMatrix.size)
         assertThat(resultData.numColumns).isEqualTo(1)
         val outValues = resultData.column(resultId).map { it ?: throw AssertionError("Should be non-null") }
         assertThat(outValues.min()).isEqualTo(4)
         assertThat(outValues.max()).isEqualTo(4)
         assertThat(outValues.size).isEqualTo(srcMatrix.size)
     }

     // Tests a supervised learning algorithm where the main and target data sets are the same
     @Test
     fun testSupervisedLearningTransformWithSingleSourceExecution() {
         val mainSource = ColumnId.create<String>("word")
         val targetSource = ColumnId.create<Boolean>("target")
         val resultId = ColumnId.create<Boolean>("prediction")

         val dg = DataGraph.build {
             val src = setSource {
                 columnIds += mainSource
                 trainOnlyColumnIds += targetSource
             }

             val trans = addTransform(src, src, TrueIfSeenTransform(mainSource, targetSource, resultId))

             result = trans
         }

         // The algorithm should learn to predict true for "foo" and "baz" but nothing else.
         val trainMatrix = listOf(
                 listOf("foo", true),
                 listOf("bar", false),
                 listOf("baz", true),
                 listOf("foo", false)
         )
         val trainData = dg.createTrainingSource(trainMatrix)

         val trainRes = dg.trainTransform(trainData, Executors.newSingleThreadExecutor()).get()

         // Expected predictions
         assertThat(trainRes.column(resultId)).containsExactly(true, false, true, true)

         // Now that it's trained we should be able to ask it to make predictions on unlabeled data.
         val testMatrix = listOf(
                 listOf("foo"),
                 listOf("bar"),
                 listOf("baz")
         )
         val testData = dg.createSource(testMatrix)

         val testRes = dg.transform(testData, Executors.newSingleThreadExecutor()).get()

         assertThat(testRes.column(resultId)).containsExactly(true, false, true)
     }

     // Tests a supervised learning algorithm where the main and target data sets are different
     @Test
     fun testSupervisedLearningTransformWithDifferentSourceExecution() {
         val mainSource = ColumnId.create<String>("word")
         val targetSource = ColumnId.create<Boolean>("target")
         val invertedTarget = ColumnId.create<Boolean>("inverted")
         val resultId = ColumnId.create<Boolean>("prediction")

         var theInverter: InvertBoolean? = null

         val dg = DataGraph.build {
             val src = setSource {
                 columnIds += mainSource
                 trainOnlyColumnIds += targetSource
             }

             theInverter = InvertBoolean(targetSource, invertedTarget)

             val inverter = addTransform(src, theInverter!!)

             val trans = addTransform(src, inverter,
                     TrueIfSeenTransform(mainSource, invertedTarget, resultId))

             result = trans
         }

         // The algorithm should learn to predict true for "foo" and "bar" but nothing else.
         val trainMatrix = listOf(
                 listOf("foo", true),
                 listOf("bar", false),
                 listOf("baz", true),
                 listOf("foo", false)
         )
         val trainData = dg.createTrainingSource(trainMatrix)

         val trainRes = dg.trainTransform(trainData, Executors.newSingleThreadExecutor()).get()

         // Expected predictions
         assertThat(trainRes.numColumns).isEqualTo(1)
         assertThat(trainRes.column(resultId)).containsExactly(true, true, false, true)
         assertThat(theInverter!!.numCalls.get()).isEqualTo(1)

         // Now that it's trained we should be able to ask it to make predictions on unlabeled data.
         val testMatrix = listOf(
                 listOf("foo"),
                 listOf("bar"),
                 listOf("baz")
         )
         val testData = dg.createSource(testMatrix)

         val testRes = dg.transform(testData, Executors.newSingleThreadExecutor()).get()

         assertThat(testRes.numColumns).isEqualTo(1)
         assertThat(testRes.column(resultId)).containsExactly(true, true, false)
         // Make sure the inverter wasn't called a 2nd time. Shouldn't be called since it's train-only.
         assertThat(theInverter!!.numCalls.get()).isEqualTo(1)
     }

     // Like testSupervisedLearningTransformWithDifferentSourceExecution but with a complex graph for the train-only
     // stuff. Here we check that even with this complex only the proper parts are executed. The graph is as follows:
     @Test
     fun testComplexTrainOnlyGraphExecution() {
         val g1 = Graph1()
         // As per comments on graph 1, items will only be predicted true if the lower case version of them is in the
         // training data with both a true and a false target. Thus, only "foo" and "bizzle" should predict true.
         val trainMatrix = listOf(
                 listOf("FOO", true),
                 listOf("foo", false),
                 listOf("bar", false),
                 listOf("bip", true),
                 listOf("baz", true),
                 listOf("biZzLE", true),
                 listOf("BIzZle", false)
         )

         // Number of threads here pretty random - just trying to test parallelism some.
         val trainRes = g1.graph.trainTransform(g1.graph.createTrainingSource(trainMatrix),
                 Executors.newFixedThreadPool(3)).get()
         assertThat(trainRes.numColumns).isEqualTo(1)
         assertThat(trainRes.column(g1.resultId)).containsExactly(true, true, false, false, false, true, true)
         assertThat(g1.invert1.numCalls.get()).isEqualTo(1)
         assertThat(g1.invert2.numCalls.get()).isEqualTo(1)

         // Now get just a prediction
         val testMatrix = listOf(
                 listOf("FoO"),
                 listOf("bar"),
                 listOf("baZ"),
                 listOf("bizzle"))
         val predictRes = g1.graph.transform(g1.graph.createSource(testMatrix), Executors.newFixedThreadPool(2)).get()
         assertThat(predictRes.numColumns).isEqualTo(1)
         assertThat(predictRes.column(g1.resultId)).containsExactly(true, false, false, true)

         // The invert nodes are both train-only and so should not have run again.
         assertThat(g1.invert1.numCalls.get()).isEqualTo(1)
         assertThat(g1.invert2.numCalls.get()).isEqualTo(1)
     }

     @Test
     fun testMergeTransformExecution() {
         val srcColId = ColumnId.create<Int>("source")

         val resultColumns = listOf(
                 ColumnId.create<Int>("t1"),
                 ColumnId.create<Int>("t2"),
                 ColumnId.create<Int>("t3")
         )

         val dg = DataGraph.build {
             val src = setSource {
                 columnIds += srcColId
             }

             // 3 parallel transforms that add 1, 2 and 3 respectively
             val t1Temp = addTransform(src, AddConstantTransform(1, src))
             val t1 = addTransform(t1Temp, ColumnSubsetTransform.build {
                 keepAndRename(srcColId, resultColumns[0])
             })

             val t2Temp = addTransform(src, AddConstantTransform(2, src))
             val t2 = addTransform(t2Temp, ColumnSubsetTransform.build {
                 keepAndRename(srcColId, resultColumns[1])
             })

             val t3Temp = addTransform(src, AddConstantTransform(3, src))
             val t3 = addTransform(t3Temp, ColumnSubsetTransform.build {
                 keepAndRename(srcColId, resultColumns[2])
             })

             val mergeDs = merge(t1, t2, t3)

             result = mergeDs
         }

         assertThat(dg.result.columns).hasSize(3)
         // assertThat(nnMergeDs.tokens.map { it.name }.toSet()).isEqualTo(resultValIds.map { it.name }.toSet())

         val resultF = dg.transform(dg.createSource(0), Executors.newFixedThreadPool(3))
         val resultData = resultF.get()
         assertThat(resultData.numRows).isEqualTo(1)
         assertThat(resultData.numColumns).isEqualTo(3)
         assertThat(resultData.value(0, resultColumns[0])).isEqualTo(1)
         assertThat(resultData.value(0, resultColumns[1])).isEqualTo(2)
         assertThat(resultData.value(0, resultColumns[2])).isEqualTo(3)
     }

     @Test
     fun testThrowingTransformCausesGraphExecutionToThrow() {
         val input = ColumnId.create<Int>("input")
         val resultId = ColumnId.create<String>("finalResult")

         val dg = DataGraph.build {
             val src = setSource {
                 columnIds += input
             }

             val trans = addTransform(src, ThrowsExceptionTransform(resultId))
             result = trans
         }

         val srcObs = dg.createSource(88)

         val transformF = dg.transform(srcObs, Executors.newSingleThreadExecutor())
         assertThatThrownBy { transformF.get() }.hasMessageContaining(ThrowsExceptionTransform.ERROR_MESSAGE)
     }

     @Test
     fun testTransformCompletingWithExceptionCausesGraphExecutionToThrow() {
         val input = ColumnId.create<Int>("input")
         val resultId = ColumnId.create<String>("finalResult")

         val dg = DataGraph.build {
             val src = setSource {
                 columnIds += input
             }

             val trans = addTransform(src, CompletesWithExceptionTransform(resultId))
             result = trans
         }

         val srcObs = dg.createSource(88)

         val transformF = dg.transform(srcObs, Executors.newSingleThreadExecutor())
         assertThatThrownBy { transformF.get() }.hasMessageContaining(CompletesWithExceptionTransform.ERROR_MESSAGE)
     }

     class ThrowsExceptionTransform(private val resultId: ColumnId<String>) : SingleDataTransform {
         companion object {
             const val ERROR_MESSAGE = "Pretending bad things happened."
         }
         override val description: TransformDescription
             get() = TransformDescription(listOf(resultId))

         override fun transform(dataSet: DataSet): CompletableFuture<DataSet> {
             throw RuntimeException(ERROR_MESSAGE)
         }
     }

     class CompletesWithExceptionTransform(private val resultId: ColumnId<String>) : SingleDataTransform {
         companion object {
             const val ERROR_MESSAGE = "Pretending bad things happened."
         }
         override val description: TransformDescription
             get() = TransformDescription(listOf(resultId))

         override fun transform(dataSet: DataSet): CompletableFuture<DataSet> {
             val result = CompletableFuture<DataSet>()
             result.completeExceptionally(RuntimeException(ERROR_MESSAGE))
             return result
         }
     }
}
