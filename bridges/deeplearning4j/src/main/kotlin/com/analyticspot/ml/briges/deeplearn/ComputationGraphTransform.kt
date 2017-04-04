package com.analyticspot.ml.briges.deeplearn

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.datatransform.SupervisedLearningTransform
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.serialization.MultiFileMixedTransform
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculatorCG
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.dataset.MultiDataSet
import java.io.OutputStream
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 * Created by oliver on 3/30/17.
 */
class ComputationGraphTransform(val net: ComputationGraph,
        val inputCols: List<List<ColumnId<*>>>, val targetSizes: List<Int>, val trainConfig: TrainingConfig)
    : SupervisedLearningTransform, MultiFileMixedTransform {

    override fun trainTransform(dataSet: DataSet, targetDs: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        // First combined the data sets so we can randomly sub-sample a validation set for early stopping.
        val combined = dataSet + targetDs

        val (validDs, trainDs) = combined.randomSubsets(trainConfig.epochValidationFrac)

        @Suppress("UNCHECKED_CAST")
        val targetCols = targetDs.columnIds.toList() as List<ColumnId<Int>>
        val trainDataIter = RandomizingMultiDataSetIterator(trainConfig.batchSize, trainDs, inputCols, targetCols,
                targetSizes)
        val validDataIter = RandomizingMultiDataSetIterator(validDs.numRows, validDs, inputCols, targetCols,
                targetSizes)

        net.init()

        val earlyStopConfig = EarlyStoppingConfiguration.Builder<ComputationGraph>()
                .epochTerminationConditions(ScoreImprovementEpochTerminationCondition(
                        trainConfig.maxEpochWithNoImprovement))
                .scoreCalculator(DataSetLossCalculatorCG(validDataIter, true))
                .build()

        return CompletableFuture.completedFuture(DataSet.build {  })
    }

    override fun transform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        throw UnsupportedOperationException()
    }


    override fun serializeBinaryData(output: OutputStream) {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    data class TrainingConfig(val batchSize: Int,
            val epochValidationFrac: Float,
            val maxEpochWithNoImprovement: Int)
}
