package com.analyticspot.ml.bridges.smile

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.metadata.CategoricalFeatureMetaData
import com.analyticspot.ml.framework.serialization.MultiFileMixedFormat
import com.fasterxml.jackson.annotation.JacksonInject
import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonProperty
import smile.classification.Classifier
import smile.classification.ClassifierTrainer
import smile.data.Attribute
import java.io.InputStream
import java.util.ArrayList

/**
 *
 * The returned data set contains the predictions. See [SmileSoftClassifier] if you would also like the posterior
 * probabilities in the output data set.
 *
 * Smile recommends the use of Xstream for serialization
 * (https://github.com/haifengl/smile#user-content-model-serialization). That serializes models as XML. The models can
 * not be serialized to JSON so we use Xstream. We use [MultiFileMixedFormat] for the serialization even though the
 * model isn't binary. However, embedding XML in JSON get horribly ugly so this seems to make more sense.
 */
class SmileClassifier : SmileClassifierBase<Classifier<DoubleArray>> {
    constructor(targetId: ColumnId<String>,
            trainerFactory: (Array<Attribute>) -> ClassifierTrainer<DoubleArray>,
            predictionCol: ColumnId<String> = ColumnId.create<String>("predicted"))
            : super(targetId, trainerFactory, predictionCol)

    @JsonCreator
    constructor(
            @JacksonInject(MultiFileMixedFormat.INJECTED_BINARY_DATA) trainModelInputStream: InputStream,
            @JsonProperty("predictionCol") predictionCol: ColumnId<String>)
            : super(trainModelInputStream, predictionCol)

    override protected fun transformConvertedData(data: Array<DoubleArray>): DataSet {
        val predictions = ArrayList<String>(data.size)
        for (row in data) {
            val intPred = trainedModel.predict(row)
            val prediction = intToTarget[intPred] ?:
                    throw IllegalStateException("$intPred was predicted but isn't a known target value")
            predictions.add(prediction)
        }
        val resultMeta = CategoricalFeatureMetaData(false, intToTarget.values.toSet())
        return DataSet.build {
            addColumn(predictionCol, predictions, resultMeta)
        }
    }
}
