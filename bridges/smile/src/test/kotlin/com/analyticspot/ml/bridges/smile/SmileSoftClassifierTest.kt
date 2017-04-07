package com.analyticspot.ml.bridges.smile

import com.analyticspot.ml.framework.datagraph.DataGraph
import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.metadata.CategoricalFeatureMetaData
import com.analyticspot.ml.framework.serialization.GraphSerDeser
import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.within
import org.testng.annotations.Test
import smile.classification.DecisionTree
import smile.data.Attribute
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

class SmileSoftClassifierTest {
    @Test
    fun testWorks() {
        val maxNodes = 20
        val dtTrainerFactory = { attrs: Array<Attribute> -> DecisionTree.Trainer(attrs, maxNodes) }
        val trainData = generateDtData()

        val treeTransform = SmileSoftClassifier(trainData.targetId, dtTrainerFactory)

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

        // 4 columns: 1 prediction and then posteriors for all 3 target values
        assertThat(trainResult.numColumns).isEqualTo(4)
        assertThat(trainResult.numRows).isEqualTo(trainData.data.numRows)

        // And just to make sure we're not just using the target value to predict lets build a DataSet that violates the
        // rule and make sure we generate the wrong prediction.
        val indColId = ColumnId.create<Boolean>("ind")
        val testDs = DataSet.build {
            addColumn(indColId, listOf(true, false))
        }

        val testResult = dg.transform(testDs, exec).get()
        assertThat(testResult.numRows).isEqualTo(2)
        assertThat(testResult.numColumns).isEqualTo(4)

        // For the first row ind was true so we expect the prediction to be A and the posteriors to be .9, .1, and 0
        assertThat(testResult.value(0, treeTransform.predictionCol)).isEqualTo("A")
        assertThat(testResult.value(0, treeTransform.classProbsId.generateId("A"))).isCloseTo(0.9, within(0.05))
        assertThat(testResult.value(0, treeTransform.classProbsId.generateId("B"))).isCloseTo(0.1, within(0.05))
        assertThat(testResult.value(0, treeTransform.classProbsId.generateId("C"))).isCloseTo(0.0, within(0.05))

        // For the 2nd row ind was false so we expect the preciction to be C
        assertThat(testResult.value(1, treeTransform.predictionCol)).isEqualTo("C")
        assertThat(testResult.value(1, treeTransform.classProbsId.generateId("A"))).isCloseTo(0.0, within(0.05))
        assertThat(testResult.value(1, treeTransform.classProbsId.generateId("B"))).isCloseTo(0.2, within(0.05))
        assertThat(testResult.value(1, treeTransform.classProbsId.generateId("C"))).isCloseTo(0.8, within(0.05))
    }

    @Test
    fun testCanSerDeser() {
        val maxNodes = 20
        val dtTrainerFactory = { attrs: Array<Attribute> -> DecisionTree.Trainer(attrs, maxNodes) }
        val trainData = generateDtData()

        val treeTransform = SmileSoftClassifier(trainData.targetId, dtTrainerFactory)

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

        dg.trainTransform(trainData.data, exec).get()

        // serialize the graph
        val output = ByteArrayOutputStream()
        val serDeser = GraphSerDeser()
        serDeser.serialize(dg, output)

        // and deserialize it
        val deserDg = serDeser.deserialize(ByteArrayInputStream(output.toByteArray()))

        // Make sure it can still make correct predictions
        val indColId = ColumnId.create<Boolean>("ind")
        val testDs = DataSet.build {
            addColumn(indColId, listOf(true, false))
        }

        val testResult = deserDg.transform(testDs, exec).get()
        assertThat(testResult.numRows).isEqualTo(2)
        assertThat(testResult.numColumns).isEqualTo(4)

        // For the first row ind was true so we expect the prediction to be A and the posteriors to be .9, .1, and 0
        assertThat(testResult.value(0, treeTransform.predictionCol)).isEqualTo("A")
        assertThat(testResult.value(0, treeTransform.classProbsId.generateId("A"))).isCloseTo(0.9, within(0.05))
        assertThat(testResult.value(0, treeTransform.classProbsId.generateId("B"))).isCloseTo(0.1, within(0.05))
        assertThat(testResult.value(0, treeTransform.classProbsId.generateId("C"))).isCloseTo(0.0, within(0.05))

        // For the 2nd row ind was false so we expect the preciction to be C
        assertThat(testResult.value(1, treeTransform.predictionCol)).isEqualTo("C")
        assertThat(testResult.value(1, treeTransform.classProbsId.generateId("A"))).isCloseTo(0.0, within(0.05))
        assertThat(testResult.value(1, treeTransform.classProbsId.generateId("B"))).isCloseTo(0.2, within(0.05))
        assertThat(testResult.value(1, treeTransform.classProbsId.generateId("C"))).isCloseTo(0.8, within(0.05))

    }

    // Return type for generateDtData below
    data class DataAndTarget(val data: DataSet, val targetId: ColumnId<String>)

    // Data has only 1 independent variable, "ind", which is boolean. If it is true the target will be "A" 90% of the
    // time, and "B" 10% of the time. If it's false the target will be "C" 80% of the time and B 20% of the time. We
    // always generate 200 instances so the above numbers come out nice and clean.
    private fun generateDtData() : DataAndTarget {
        val indValues = mutableListOf<Boolean>()
        val targetValues = mutableListOf<String>()

        // When ind is true we get A 90% of the time and B 10% of the time
        indValues.addAll((1..100).map { true })
        targetValues.addAll((1..90).map { "A" })
        targetValues.addAll((1..10).map { "B" })

        check(indValues.size == 100)
        check(targetValues.size == 100)

        // When ind is false we get C 80% of the time and B 20% of the time
        indValues.addAll((1..100).map { false })
        targetValues.addAll((1..80).map { "C" })
        targetValues.addAll((1..20).map { "B" })
        check(indValues.size == 200)
        check(targetValues.size == 200)

        val targetId = ColumnId.create<String>("Target")
        val targetMeta = CategoricalFeatureMetaData(false, setOf("A", "B", "C"))

        val data = DataSet.build {
            addColumn(ColumnId.create<Boolean>("ind"), indValues)
            addColumn(targetId, targetValues, targetMeta)
        }

        return DataAndTarget(data, targetId)
    }
}
