package com.analyticspot.ml.briges.deeplearn

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.utils.DataUtils
import com.google.common.util.concurrent.MoreExecutors
import org.assertj.core.api.Assertions.assertThat
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration
import org.deeplearning4j.nn.conf.LearningRatePolicy
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory
import org.testng.annotations.Test
import java.util.Random

class ComputationGraphTransformTest {
    companion object {
        private val log = LoggerFactory.getLogger(ComputationGraphTransformTest::class.java)
    }

    @Test
    fun testCanTrainSimpleMlp() {
        val targetColId = ColumnId.create<String>("target")
        val encodedTargetColId = ColumnId.create<Int>("target")
        val ds = DataSet.fromSaved(javaClass.getResourceAsStream("/iris.data.json"))

        val (trainDs, validDs) = ds.randomSubsets(0.75f, Random(222))
        log.info("Training data has {} rows. Validation data has {} rows.", trainDs.numRows, validDs.numRows)

        val trainFeatures = trainDs.allColumnsExcept("target")
        val (trainTargetsCols, targetMapping) = DataUtils.encodeCategorical(trainDs.column(targetColId))
        val trainTargets = DataSet.build {
            addColumn(encodedTargetColId, trainTargetsCols)
        }

        // Build a simple computation graph (in fact, we could have used the simpler `MultiLayerConfiguration` for this
        // but that's not what we're testing) that has a single hidden layer with 4 units.
        log.info("Building computation ComputationGraphConfig")
        val compGraphConfig: ComputationGraphConfiguration = NeuralNetConfiguration.Builder()
                .learningRate(0.01)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.RMSPROP)
                .graphBuilder()
                .addInputs("input")
                .addLayer("l1", DenseLayer.Builder()
                        .nIn(trainFeatures.numColumns)
                        .nOut(10)
                        .activation(Activation.TANH)
                        .build(), "input")
                .addLayer("output", OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(10)
                        .nOut(targetMapping.size)
                        .build(), "l1")
                .setOutputs("output")
                .build()

        log.info("Constructing ComputationGraph from the configuration.")
        val nn = ComputationGraph(compGraphConfig)

        val colNamer = { output: Int, targetVal: Int ->
            assertThat(output).isEqualTo(0)
            "Posterior-${targetVal}"

        }

        val transform = ComputationGraphTransform.build {
            net = nn
            inputCols = listOf(trainFeatures.columnIds.toList())
            targetSizes = listOf(3)
            outColNameGenerator = colNamer
        }

        log.info("Starting to train network")
        transform.trainTransform(trainFeatures, trainTargets, MoreExecutors.newDirectExecutorService()).get()

        // Now assess accuracy. If things are working we should have a very high accuracy rate on the validation data
        val posteriors = transform.transform(validDs, MoreExecutors.newDirectExecutorService()).get()
        assertThat(posteriors.numRows).isEqualTo(validDs.numRows)

        // For each row, find the column with the max posterior. Then map that back to the string version of the target
        val predictedTargets: List<String> = 0.until(posteriors.numRows).map { rowIdx ->
            0.until(posteriors.numColumns).maxBy { colIdx ->
                posteriors.value(rowIdx, ColumnId.create<Double>(colNamer (0, colIdx)))!! }
        }.map { targetMapping[it]!! }

        assertThat(predictedTargets).hasSize(validDs.numRows)

        val numCorrect = predictedTargets.zip(validDs.column(targetColId)).filter {
            it.first == it.second
        }.count()
        val accuracy = numCorrect.toDouble()/validDs.numRows.toDouble()
        log.info("Model accuracy on validation set: {}", accuracy)

        assertThat(accuracy).isGreaterThan(0.7)
    }
}
