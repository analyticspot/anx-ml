package com.analyticspot.ml.bridges.smile

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.description.ColumnIdGroup
import com.analyticspot.ml.framework.metadata.CategoricalFeatureMetaData
import com.analyticspot.ml.framework.metadata.MaybeMissingMetaData
import com.analyticspot.ml.framework.serialization.MultiFileMixedFormat
import com.fasterxml.jackson.annotation.JacksonInject
import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonProperty
import smile.classification.ClassifierTrainer
import smile.classification.SoftClassifier
import smile.data.Attribute
import java.io.InputStream
import java.util.ArrayList

/**
 * Like [SmileClassifier] for for "soft" classifiers. In Smile parlance a "soft classifier" is one that can provide
 * posterior probabilities in addition to a prediction. Note that if you don't want the posteriors you can use the
 * same classifier with [SmileClassifier] and the extra work of generating posteriors will simply be skipped.
 */
class SmileSoftClassifier : SmileClassifierBase<SoftClassifier<DoubleArray>> {
    /**
     * For each target value there will be a column that will contain the posterier probability of that column.
     */
    val classProbsId: ColumnIdGroup<Double>

    constructor(targetId: ColumnId<String>,
            trainerFactory: (Array<Attribute>) -> ClassifierTrainer<DoubleArray>,
            predictionCol: ColumnId<String> = ColumnId.create<String>("predicted"),
            classProbsId: ColumnIdGroup<Double> = ColumnIdGroup.create("prob"))
            : super(targetId, trainerFactory, predictionCol) {
        this.classProbsId = classProbsId
    }

    @JsonCreator
    constructor(
            @JacksonInject(MultiFileMixedFormat.INJECTED_BINARY_DATA) trainModelInputStream: InputStream,
            @JsonProperty("predictionCol") predictionCol: ColumnId<String>,
            @JsonProperty("classProbsId") classProbsId: ColumnIdGroup<Double>)
            : super(trainModelInputStream, predictionCol) {
        this.classProbsId = classProbsId
    }

    override fun transformConvertedData(data: Array<DoubleArray>): DataSet {
        val predictions = ArrayList<String>(data.size)
        val probs = Array<ArrayList<Double>>(intToTarget.size) {
            ArrayList(data.size)
        }
        for (row in data) {
            val curProbs = DoubleArray(intToTarget.size)
            val intPred = trainedModel.predict(row, curProbs)
            val prediction = intToTarget[intPred] ?:
                    throw IllegalStateException("$intPred was predicted but isn't a known target value")
            predictions.add(prediction)
            // Smile gives us an array of probabilities but we're going to need column arrays so here we append the
            // value (posterior probability) for each target into a separate array.
            intToTarget.keys.forEach {
                probs[it].add(curProbs[it])
            }
        }

        val resultMeta = CategoricalFeatureMetaData(false, intToTarget.values.toSet())
        val probMeta = MaybeMissingMetaData(false)
        return DataSet.build {
            addColumn(predictionCol, predictions, resultMeta)
            probs.forEachIndexed { targetIdx, probs ->
                addColumn(classProbsId.generateId(intToTarget[targetIdx]!!), probs, probMeta)
            }
        }
    }
}
