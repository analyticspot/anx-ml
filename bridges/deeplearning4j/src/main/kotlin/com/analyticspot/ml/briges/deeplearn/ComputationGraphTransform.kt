package com.analyticspot.ml.briges.deeplearn

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.datatransform.SupervisedLearningTransform
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.description.ColumnIdGroup
import com.analyticspot.ml.framework.metadata.MaybeMissingMetaData
import com.analyticspot.ml.framework.serialization.MultiFileMixedTransform
import com.fasterxml.jackson.annotation.JsonIgnore
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.EarlyStoppingResult
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculatorCG
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.api.ndarray.INDArray
import org.slf4j.LoggerFactory
import java.io.OutputStream
import java.util.ArrayList
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 * Created by oliver on 3/30/17.
 */
class ComputationGraphTransform : SupervisedLearningTransform, MultiFileMixedTransform {

    @get:JsonIgnore
    private val net: ComputationGraph

    private val inputCols: List<List<ColumnId<*>>>

    private val targetSizes: List<Int>

    val outColPosteriorGroups: List<ColumnIdGroup<Double>>

    val outColPredictions: List<ColumnId<Int>>

    // Is null when deserialized after training. We only need the training config to learn
    var config: Builder? = null

    private constructor(config: Builder) {
        require(config.inputCols.size > 0)
        require(config.batchSize > 0)
        require(config.epochValidationFrac >= 0.0 && config.epochValidationFrac < 1.0)
        require(config.maxEpochWithNoImprovement > 0)
        require(config.targetSizes.size > 0)
        require(config.outColPosteriorGroups.size == config.targetSizes.size)
        require(config.outColPredictions.size == config.targetSizes.size)

        this.config = config
        net = config.net
        inputCols = config.inputCols
        targetSizes = config.targetSizes
        outColPosteriorGroups = config.outColPosteriorGroups
        outColPredictions = config.outColPredictions
    }

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

        val (validDs, trainDs) = combined.randomSubsets(config!!.epochValidationFrac)

        @Suppress("UNCHECKED_CAST")
        val targetCols = targetDs.columnIds.toList() as List<ColumnId<Int>>
        val trainDataIter = RandomizingMultiDataSetIterator(
                config!!.batchSize, trainDs, inputCols, targetCols, targetSizes)
        val validDataIter = RandomizingMultiDataSetIterator(
                validDs.numRows, validDs, inputCols, targetCols, targetSizes)

        net.init()

        val earlyStopConfig = EarlyStoppingConfiguration.Builder<ComputationGraph>()
                .epochTerminationConditions(ScoreImprovementEpochTerminationCondition(
                        config!!.maxEpochWithNoImprovement))
                .scoreCalculator(DataSetLossCalculatorCG(validDataIter, true))
                .build()

        val earlyStoppingTrainer = EarlyStoppingGraphTrainer(
                earlyStopConfig, net, trainDataIter, ReportingTrainListener())

        log.info("Starting NN training.")
        earlyStoppingTrainer.fit()
        log.info("Training complete.")

        return transform(dataSet, exec)
    }

    override fun transform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        val mds = Utils.toMultiDataSet(dataSet, inputCols, listOf(), listOf())
        check(mds.features[0].rows() == dataSet.numRows)
        val posteriors = net.output(*mds.features)
        // Predictions should now be an array of matrices. Each matrix has 1 row per row in dataSet and 1 column per
        // target value. The value at row i, column j for that matrix is the posterior probability that example i has
        // target value j. There is one such matrix for each output in our computation graph.
        check(posteriors.size == targetSizes.size)
        check(posteriors[0].shape()[0] == dataSet.numRows)
        return CompletableFuture.completedFuture(predictionsToDataSet(posteriors))
    }

    private fun predictionsToDataSet(predictions: Array<INDArray>): DataSet {
        // One INDArray for each output
        require(predictions.size == targetSizes.size)
        return DataSet.build {
            for (outIdx in 0.until(targetSizes.size)) {
                check(predictions[outIdx].columns() == targetSizes[outIdx])
                val numRows = predictions[outIdx].rows()
                // This will end up being the data that holds the prediction. We're building column-wise so we
                // have a list of predictions - one for each row. Here we store the largest-to-date posterior plus
                // the target value to which that posterior corresponds.
                var mostLikely = ArrayList<Pair<Double, Int>>(numRows)
                repeat(numRows) { mostLikely.add(Pair(0.0, -1)) }

                for (posteriorIdx in 0.until(targetSizes[outIdx])) {
                    val colId = outColPosteriorGroups[outIdx].generateId(posteriorIdx.toString())
                    val colData = ArrayList<Double>(numRows)
                    0.until(numRows).forEach {
                        val curPosterior = predictions[outIdx].getDouble(it, posteriorIdx)
                        colData.add(curPosterior)
                        if (curPosterior > mostLikely[it].first) {
                            mostLikely[it] = Pair(curPosterior, posteriorIdx)
                        }
                    }
                    addColumn(colId, colData, MaybeMissingMetaData(false))
                }
                addColumn(outColPredictions[outIdx], mostLikely.map { it.second }, MaybeMissingMetaData(false))
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
        var batchSize: Int = 128
        var epochValidationFrac: Float = 0.1f
        var maxEpochWithNoImprovement: Int = 4

        /**
         * The [ColumnIdGroup] to use for the posteriors for each output. So the posteriors probability that output `i`
         * is class `j` will be in column `outColPosteriorGroups[i].generateId(j.toString())`.
         */
        var outColPosteriorGroups: List<ColumnIdGroup<Double>> = listOf(ColumnIdGroup.create("posterior"))

        /**
         * The [ColumnId] for the prediction (the value with the max posterior) for each output.
         */
        var outColPredictions: List<ColumnId<Int>> = listOf(ColumnId.create<Int>("prediction"))
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
