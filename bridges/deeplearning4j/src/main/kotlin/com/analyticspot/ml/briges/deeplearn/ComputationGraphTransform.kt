package com.analyticspot.ml.briges.deeplearn

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.datatransform.SupervisedLearningTransform
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.serialization.MultiFileMixedTransform
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.EarlyStoppingResult
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculatorCG
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.MultiDataSet
import org.slf4j.LoggerFactory
import java.io.OutputStream
import java.util.ArrayList
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 * Created by oliver on 3/30/17.
 */
class ComputationGraphTransform(val config: Builder)
    : SupervisedLearningTransform, MultiFileMixedTransform {

    companion object {
        val log = LoggerFactory.getLogger(ComputationGraphTransform::class.java)

        fun build(init: Builder.() -> Unit): ComputationGraphTransform {
            val bldr = Builder()
            bldr.init()
            return ComputationGraphTransform(bldr)
        }
    }

    override fun trainTransform(dataSet: DataSet, targetDs: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        // First combined the data sets so we can randomly sub-sample a validation set for early stopping.
        val combined = dataSet + targetDs

        val (validDs, trainDs) = combined.randomSubsets(config.epochValidationFrac)

        @Suppress("UNCHECKED_CAST")
        val targetCols = targetDs.columnIds.toList() as List<ColumnId<Int>>
        val trainDataIter = RandomizingMultiDataSetIterator(
                config.batchSize, trainDs, config.inputCols, targetCols, config.targetSizes)
        val validDataIter = RandomizingMultiDataSetIterator(
                validDs.numRows, validDs, config.inputCols, targetCols, config.targetSizes)

        config.net.init()

        val earlyStopConfig = EarlyStoppingConfiguration.Builder<ComputationGraph>()
                .epochTerminationConditions(ScoreImprovementEpochTerminationCondition(
                        config.maxEpochWithNoImprovement))
                .scoreCalculator(DataSetLossCalculatorCG(validDataIter, true))
                .build()

        val earlyStoppingTrainer = EarlyStoppingGraphTrainer(earlyStopConfig, config.net, trainDataIter,
                ReportingTrainListener())

        return CompletableFuture.completedFuture(DataSet.build {  })
    }

    override fun transform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        val mds = Utils.toMultiDataSet(dataSet, config.inputCols, listOf(), listOf())
        val predictions = config.net.output(*mds.features)
        return CompletableFuture.completedFuture(predictionsToDataSet(predictions))
    }

    private fun predictionsToDataSet(predictions: Array<INDArray>): DataSet {
        val numTotalPredictions = config.targetSizes.sum()
        require(predictions[0].columns() == numTotalPredictions)
        var curPredCol = 0
        return DataSet.build {
            for (outIdx in 0.until(config.targetSizes.size)) {
                for (pIdx in 0.until(config.targetSizes[outIdx])) {
                    check(curPredCol < numTotalPredictions)

                    val colName = config.outColNameGenerator(outIdx, pIdx)
                    val colData = ArrayList<Double>(predictions.size)
                    predictions.forEach { predRow -> colData.add(predRow.getDouble(curPredCol))  }
                    ++curPredCol
                }
            }
        }
    }


    override fun serializeBinaryData(output: OutputStream) {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    class Builder() {
        lateinit var net: ComputationGraph
        lateinit var inputCols: List<List<ColumnId<*>>>
        lateinit var targetSizes: List<Int>
        var batchSize: Int = 100
        var epochValidationFrac: Float = 0.1f
        var maxEpochWithNoImprovement: Int = 4
        /**
         * A function that generates column names for the predictions. The inputs are the output layer (output layer i
         * is making predictions for the i'th column of the `trainData` passed to [trainTransform]) and the target value
         * to which the posterior applies. In other words, `outColNameGenerator(i, j)` tells you the posterior
         * probability that the i^th column of `trainData` had value `j`.
         */
        var outColNameGenerator: (output: Int, category: Int) -> String = { output, category ->
            "Out${output}Posterior${category}"
        }
    }

    private class ReportingTrainListener : EarlyStoppingListener<ComputationGraph> {
        override fun onCompletion(esResult: EarlyStoppingResult<ComputationGraph>) {
            log.info("Training complete: {}", esResult)
        }

        override fun onEpoch(epochNum: Int, score: Double,
                esConfig: EarlyStoppingConfiguration<ComputationGraph>, net: ComputationGraph) {
            log.info("Completed training epoch {}. Score: {}", epochNum, score)
        }

        override fun onStart(esConfig: EarlyStoppingConfiguration<ComputationGraph>, net: ComputationGraph) {
            log.info("Starting to train network.")
        }

    }

}
