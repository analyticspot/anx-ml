package com.analyticspot.ml.briges.deeplearn

import com.analyticspot.ml.framework.datagraph.DataGraph
import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.description.ColumnIdGroup
import com.analyticspot.ml.framework.serialization.GraphSerDeser
import com.analyticspot.ml.framework.utils.DataUtils
import com.google.common.util.concurrent.MoreExecutors
import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.within
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration
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
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.util.Random

// Some of these tests use the Iris dataset. That's a fairly famous ML example data set. Info can be found here:
// https://archive.ics.uci.edu/ml/datasets/Iris
// I downloaded that and with some command line junk converted it to our DataSet format. I also scaled all the inputs
// to 0 mean and unit variance.
class ComputationGraphTransformTest : Dl4jTestBase() {
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
                .learningRate(0.5)
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

        val transform = ComputationGraphTransform.build {
            net = nn
            inputCols = listOf(trainFeatures.columnIds.toList())
            targetSizes = listOf(encodedTargetColId to 3)
            trainingParams = ComputationGraphTransform.Builder.TrainingParams.build {
                // 100 iterations gives really good performance. It can get better but that's slow and we don't care.
                maxEpochs = 100
            }
        }

        log.info("Starting to train network")
        transform.trainTransform(trainFeatures, trainTargets, MoreExecutors.newDirectExecutorService()).get()

        // Now assess accuracy. If things are working we should have a very high accuracy rate on the validation data
        val resultDs = transform.transform(validDs, MoreExecutors.newDirectExecutorService()).get()
        assertThat(resultDs.numRows).isEqualTo(validDs.numRows)

        // For each row, find the column with the max posterior. This should be equal to the prediction for that row.
        0.until(resultDs.numRows).forEach { rowIdx ->
            val maxPosteriorIdx = 0.until(targetMapping.size).maxBy { colIdx ->
                val colId = transform.outColPosteriorGroups[0].generateId(colIdx.toString())
                resultDs.value(rowIdx, colId)!!
            }
            assertThat(maxPosteriorIdx).isEqualTo(resultDs.value(rowIdx, transform.outColPredictions[0]))
        }

        // Map the targets to their string values
        val predictedTargets = resultDs.column(transform.outColPredictions[0]).map { targetMapping[it]!! }
        assertThat(predictedTargets).hasSize(validDs.numRows)

        val numCorrect = predictedTargets.zip(validDs.column(targetColId)).filter {
            it.first == it.second
        }.count()
        val accuracy = numCorrect.toDouble() / validDs.numRows.toDouble()
        log.info("Model accuracy on validation set: {}", accuracy)

        assertThat(accuracy).isGreaterThan(0.7)
    }

    // A ComputationGraph need not be a fully connected MLP, it can have multiple inputs and multiple outputs. This
    // tests that that works as expected. To do that we re-use the Iris data set but split the inputs into 2 sets:
    // One is all the input features and the other is just the petal features. We use give the petal-only features
    // their own hidden layer and output. The output has a boolean target which is 1 if the "full target is
    // "Iris-virginica" and 0 otherwise. These two hidden layers are then combined with yet another one and
    // the normal output.
    @Test
    fun testMultipleInputsAndOutputs() {
        val targetColId = ColumnId.create<String>("target")
        val encodedTargetColId = ColumnId.create<Int>("target")
        val isVirginicaColId = ColumnId.create<Int>("isVirginica")
        val ds = DataSet.fromSaved(javaClass.getResourceAsStream("/iris.data.json"))

        val (trainDs, validDs) = ds.randomSubsets(0.75f, Random(222))
        log.info("Training data has {} rows. Validation data has {} rows.", trainDs.numRows, validDs.numRows)

        val trainFeatures = trainDs.allColumnsExcept("target")
        val (trainTargetsCols, targetMapping) = DataUtils.encodeCategorical(trainDs.column(targetColId))
        val trainTargets = DataSet.build {
            addColumn(encodedTargetColId, trainTargetsCols)
            addColumn(isVirginicaColId, trainDs.column(targetColId).mapToColumn {
                if (it == "Iris-virginica") 1 else 0
            })
        }

        // Build a simple computation graph (in fact, we could have used the simpler `MultiLayerConfiguration` for this
        // but that's not what we're testing) that has a single hidden layer with 4 units.
        log.info("Building computation ComputationGraphConfig")
        val compGraphConfig: ComputationGraphConfiguration = NeuralNetConfiguration.Builder()
                .learningRate(0.5)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.RMSPROP)
                .graphBuilder()
                .addInputs("allFeatures", "justPetals")
                .addLayer("l1", DenseLayer.Builder()
                        .nIn(4)
                        .nOut(10)
                        .activation(Activation.TANH)
                        .build(), "allFeatures")
                .addLayer("petalHidden", DenseLayer.Builder()
                        .nIn(2)
                        .nOut(4)
                        .activation(Activation.TANH)
                        .build(), "justPetals")
                .addLayer("isVirginica", OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(4)
                        .nOut(2)
                        .build(), "petalHidden")
                .addLayer("output", OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(14)
                        .nOut(targetMapping.size)
                        .build(), "l1", "petalHidden")
                .setOutputs("output", "isVirginica")
                .build()

        log.info("Constructing ComputationGraph from the configuration.")
        val nn = ComputationGraph(compGraphConfig)

        val transform = ComputationGraphTransform.build {
            net = nn

            inputCols = listOf(
                    trainFeatures.columnIds.toList(),
                    listOf(
                            trainFeatures.columnIdWithName<Double>("PetalLength"),
                            trainFeatures.columnIdWithName<Double>("PetalWidth")))

            targetSizes = listOf(encodedTargetColId to 3, isVirginicaColId to 2)

            outColPosteriorGroups = listOf(ColumnIdGroup.create<Double>("posterior"),
                    ColumnIdGroup.create<Double>("isVirginicaPosterior"))

            outColPredictions = listOf(ColumnId.create<Int>("prediction"), ColumnId.create<Int>("isVirginicaPred"))

            trainingParams = ComputationGraphTransform.Builder.TrainingParams.build {
                // 100 iterations gives really good performance. It can get better but that's slow and we don't care.
                maxEpochs = 100
            }
        }

        log.info("Starting to train network")
        transform.trainTransform(trainFeatures, trainTargets, MoreExecutors.newDirectExecutorService()).get()

        // Now assess accuracy. If things are working we should have a very high accuracy rate on the validation data
        val resultDs = transform.transform(validDs, MoreExecutors.newDirectExecutorService()).get()
        assertThat(resultDs.numRows).isEqualTo(validDs.numRows)

        // For each row, find the column with the max posterior. This should be equal to the prediction for that row.
        0.until(resultDs.numRows).forEach { rowIdx ->
            val maxPosteriorIdx = 0.until(targetMapping.size).maxBy { colIdx ->
                val colId = transform.outColPosteriorGroups[0].generateId(colIdx.toString())
                resultDs.value(rowIdx, colId)!!
            }
            assertThat(maxPosteriorIdx).isEqualTo(resultDs.value(rowIdx, transform.outColPredictions[0]))
        }

        // Map the targets to their string values
        val predictedTargets = resultDs.column(transform.outColPredictions[0]).map { targetMapping[it]!! }
        assertThat(predictedTargets).hasSize(validDs.numRows)

        val numCorrect = predictedTargets.zip(validDs.column(targetColId)).filter {
            it.first == it.second
        }.count()
        val accuracy = numCorrect.toDouble() / validDs.numRows.toDouble()
        log.info("Model accuracy on validation set: {}", accuracy)

        assertThat(accuracy).isGreaterThan(0.7)

        // For the other set of outputs, the isVirginica outputs, the posteriors should agree with the prediction and
        // the accuracy should be decent.
        0.until(resultDs.numRows).forEach { rowIdx ->
            val trueColId = transform.outColPosteriorGroups[1].generateId("1")
            val falseColId = transform.outColPosteriorGroups[1].generateId("0")
            val posteriorTrue = resultDs.value(rowIdx, trueColId)!!
            val posteriorFalse = resultDs.value(rowIdx, falseColId)!!
            assertThat(posteriorTrue + posteriorFalse).isCloseTo(1.0, within(0.00001))
            if (posteriorTrue >= 0.5) {
                assertThat(resultDs.value(rowIdx, transform.outColPredictions[1])).isEqualTo(1)
            } else {
                assertThat(resultDs.value(rowIdx, transform.outColPredictions[1])).isEqualTo(0)
            }
        }

        val numCorrectVirginica = resultDs.column(transform.outColPredictions[1]).zip(validDs.column(targetColId))
                .filter {
                    if (it.first == 1 && it.second == "Iris-virginica") {
                        true
                    } else {
                        it.first == 0 && it.second != "Iris-virginica"
                    }
                }.count()
        val isVirginicaAcc = numCorrectVirginica.toDouble() / validDs.numRows.toDouble()
        log.info("Accuracy on isVirginica output: {}", isVirginicaAcc)
        assertThat(isVirginicaAcc).isGreaterThan(0.7)
    }

    @Test
    fun testCanSerDeser() {
        val targetColId = ColumnId.create<String>("target")
        val encodedTargetColId = ColumnId.create<Int>("target")
        val ds = DataSet.fromSaved(javaClass.getResourceAsStream("/iris.data.json"))

        val trainFeatures = ds.allColumnsExcept("target")
        val (trainTargetsCols, targetMapping) = DataUtils.encodeCategorical(ds.column(targetColId))
        val trainTargets = DataSet.build {
            addColumn(encodedTargetColId, trainTargetsCols)
        }
        val trainDs = trainFeatures + trainTargets

        // Build a simple computation graph (in fact, we could have used the simpler `MultiLayerConfiguration` for this
        // but that's not what we're testing) that has a single hidden layer with 4 units.
        log.info("Building computation ComputationGraphConfig")
        val compGraphConfig: ComputationGraphConfiguration = NeuralNetConfiguration.Builder()
                .learningRate(0.5)
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

        val transform = ComputationGraphTransform.build {
            net = nn
            inputCols = listOf(trainFeatures.columnIds.toList())
            targetSizes = listOf(encodedTargetColId to 3)
            trainingParams = ComputationGraphTransform.Builder.TrainingParams.build {
                // Make the test fast. We don't care too much about accuracy.
                maxEpochs = 4
            }
        }

        val dg = DataGraph.build {
            val src = dataSetSource()
            val targ = keepColumns(src, encodedTargetColId)
            val features = removeColumns(src, encodedTargetColId)
            val trans = addTransform(features, targ, transform)
            result = trans
        }

        log.info("Starting to train network")
        dg.trainTransform(trainDs, MoreExecutors.newDirectExecutorService()).get()

        // Now serialize
        val out = ByteArrayOutputStream()
        val sds = GraphSerDeser()
        log.debug("Saving DataGraph")
        sds.serialize(dg, out)

        log.debug("Loading saved DataGraph")
        val dgDeser = sds.deserialize(ByteArrayInputStream(out.toByteArray()))

        // We don't care about accuracy here but we do care that it works.
        val resultDs = dgDeser.transform(trainDs, MoreExecutors.newDirectExecutorService()).get()

        assertThat(resultDs.numRows).isEqualTo(trainDs.numRows)
        // 1 column for each posterior plus the prediction itself.
        assertThat(resultDs.numColumns).isEqualTo(4)
    }
}
