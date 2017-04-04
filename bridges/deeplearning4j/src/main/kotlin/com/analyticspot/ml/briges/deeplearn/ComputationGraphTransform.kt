package com.analyticspot.ml.briges.deeplearn

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.datatransform.SupervisedLearningTransform
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.metadata.MaybeMissingMetaData
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

        val earlyStoppingTrainer = EarlyStoppingGraphTrainer(
                earlyStopConfig, config.net, trainDataIter, ReportingTrainListener())

        log.info("Starting NN training.")
        earlyStoppingTrainer.fit()
        log.info("Training complete.")

        return transform(dataSet, exec)
    }

    override fun transform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        val mds = Utils.toMultiDataSet(dataSet, config.inputCols, listOf(), listOf())
        check(mds.features[0].rows() == dataSet.numRows)
        val posteriors = config.net.output(*mds.features)
        // Predictions should now be an array of matrices. Each matrix has 1 row per row in dataSet and 1 column per
        // target value. The value at row i, column j for that matrix is the posterior probability that example i has
        // target value j. There is one such matrix for each output in our computation graph.
        check(posteriors.size == config.targetSizes.size)
        check(posteriors[0].shape()[0] == dataSet.numRows)
        return CompletableFuture.completedFuture(predictionsToDataSet(posteriors))
    }

    private fun predictionsToDataSet(predictions: Array<INDArray>): DataSet {
        // One INDArray for each output
        require(predictions.size == config.targetSizes.size)
        return DataSet.build {
            var curPredCol = 0
            for (outIdx in 0.until(config.targetSizes.size)) {
                check(predictions[outIdx].columns() == config.targetSizes[outIdx])
                for (posteriorIdx in 0.until(config.targetSizes[outIdx])) {
                    val colName = config.outColNameGenerator(outIdx, posteriorIdx)
                    val colData = ArrayList<Double>(predictions[outIdx].rows())
                    0.until(predictions[outIdx].rows()).forEach {
                        colData.add(predictions[outIdx].getDouble(it, posteriorIdx))
                    }
                    addColumn(ColumnId.create<Double>(colName), colData, MaybeMissingMetaData(false))
                    ++curPredCol
                }
            }
        }
    }


    override fun serializeBinaryData(output: OutputStream) {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    class Builder() {
        /**
         * We will call `init` on the graph so the user need not do that.
         */
        lateinit var net: ComputationGraph
        lateinit var inputCols: List<List<ColumnId<*>>>
        lateinit var targetSizes: List<Int>
        var batchSize: Int = 32
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

    // This doesn't do much but an instance of this is required. Dl4j already does a bunch of logging so there's not
    // much to add here.
    private class ReportingTrainListener : EarlyStoppingListener<ComputationGraph> {
        override fun onCompletion(esResult: EarlyStoppingResult<ComputationGraph>) {
            log.info("Training complete: {}", esResult)
        }

        override fun onEpoch(epochNum: Int, score: Double,
                esConfig: EarlyStoppingConfiguration<ComputationGraph>, net: ComputationGraph) {
        }

        override fun onStart(esConfig: EarlyStoppingConfiguration<ComputationGraph>, net: ComputationGraph) {
            log.info("Starting to train network.")
        }

    }

}
