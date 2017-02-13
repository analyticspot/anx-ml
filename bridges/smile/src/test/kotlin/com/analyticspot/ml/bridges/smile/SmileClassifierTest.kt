package com.analyticspot.ml.bridges.smile

import com.analyticspot.ml.framework.datagraph.DataGraph
import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.metadata.CategoricalFeatureMetaData
import kotlinx.support.jdk8.streams.toList
import org.assertj.core.api.Assertions.assertThat
import org.slf4j.LoggerFactory
import org.testng.annotations.Test
import smile.classification.DecisionTree
import smile.data.Attribute
import java.util.Random
import java.util.concurrent.Executors

class SmileClassifierTest {
    companion object {
        private val log = LoggerFactory.getLogger(SmileClassifierTest::class.java)
    }

    @Test
    fun testSimpleClassifier() {
        val maxNodes = 20
        val dtTrainerFactory = { attrs: Array<Attribute> -> DecisionTree.Trainer(attrs, maxNodes) }
        // 400 rows of data. The 2nd argument is just a random seed
        val trainData = generateDtData(400, 0)

        val treeTransform = SmileClassifier(trainData.targetId, dtTrainerFactory)

        val dg = DataGraph.build {
            val src = setSource {
                columnIds += trainData.data.columnIds.filter { it.name != trainData.targetId.name }
                trainOnlyColumnIds += trainData.targetId
            }

            val inputSet = removeColumns(src, trainData.targetId)

            val targetSet = keepColumns(src, trainData.targetId)

            val tree = addTransform(inputSet, targetSet, treeTransform)

            result = tree
        }

        val exec = Executors.newFixedThreadPool(2)

        val trainResult = dg.trainTransform(trainData.data, exec).get()

        assertThat(trainResult.numColumns).isEqualTo(1)
        assertThat(trainResult.numRows).isEqualTo(trainData.data.numRows)

        val predColumn = trainResult.columnIds.first()

        for (i in 0 until trainData.data.numRows) {
            val prediction = trainResult.value(i, predColumn)
            val target = trainData.data.value(i, trainData.targetId)
            assertThat(prediction).isEqualTo(target)
        }

        // And just to make sure we're not just using the target value to predict lets build a DataSet that violates the
        // rule and make sure we generate the wrong prediction.
        val wrongDs = DataSet.build {
            addColumn(trainData.data.columnIdWithName("A"), listOf("A2"), trainData.data.metaData["A"])
            addColumn(trainData.data.columnIdWithName("B"), listOf(0.2))
            addColumn(trainData.data.columnIdWithName("C"), listOf(true))
            addColumn(trainData.data.columnIdWithName("Target"), listOf("T3"), trainData.data.metaData["Target"])
        }

        val wrongResult = dg.transform(wrongDs, exec).get()
        assertThat(wrongResult.numRows).isEqualTo(1)
        assertThat(wrongResult.numColumns).isEqualTo(1)
        // We followed the T1 rule so that should have been the prediction even though we passed in a target of T3
        assertThat(wrongResult.value(0, predColumn)).isEqualTo("T1")
    }

    // Return type for generateDtData below
    data class DataAndTarget(val data: DataSet, val targetId: ColumnId<String>)

    // Generates some data that a decision tree should be able to classify perfectly. It contains 3 variables and a
    // target. The variables are A, a categofical with values from [A1, A2, A3, A4], B, a numeric with values in the
    // range [0, 1), and C a boolean. The target is a deterministic function of A, B, and C (see the code for the
    // function).
    private fun generateDtData(numRows: Long, seed: Long): DataAndTarget {
        // Use fixed seed so the test is deterministic
        val rng = Random(seed)

        // A
        val possibleAValues = listOf("A1", "A2", "A3", "A4")
        val a = rng.ints(0, possibleAValues.size).mapToObj { possibleAValues[it] }.limit(numRows).toList()

        // B
        val b = rng.doubles(0.0, 1.0).limit(numRows).toList()

        // C
        val c = rng.ints(0, 2).mapToObj { it == 1 }.limit(numRows).toList()

        // Target
        val target = a.indices.map { idx ->
            if (a[idx] == "A2" && c[idx]) {
                "T1"
            } else if (b[idx] > 0.5) {
                "T2"
            } else {
                "T3"
            }
        }

        val targetId = ColumnId.create<String>("Target")
        val targetMeta = CategoricalFeatureMetaData(false, setOf("T1", "T2", "T3"))

        val data = DataSet.build {
            addColumn(ColumnId.create<String>("A"), a, CategoricalFeatureMetaData(false, possibleAValues.toSet()))
            addColumn(ColumnId.create<Double>("B"), b)
            addColumn(ColumnId.create<Boolean>("C"), c)
            addColumn(targetId, target, targetMeta)
        }

        return DataAndTarget(data, targetId)
    }
}
