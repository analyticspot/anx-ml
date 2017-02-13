package com.analyticspot.ml.bridges.smile

import com.analyticspot.ml.framework.dataset.Column
import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.datatransform.TargetSupervisedLearningTransform
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.metadata.CategoricalFeatureMetaData
import com.analyticspot.ml.framework.serialization.MultiFileMixedFormat
import com.analyticspot.ml.framework.serialization.MultiFileMixedTransform
import com.fasterxml.jackson.annotation.JacksonInject
import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonIgnore
import com.fasterxml.jackson.annotation.JsonProperty
import com.thoughtworks.xstream.XStream
import org.slf4j.LoggerFactory
import smile.classification.Classifier
import smile.classification.ClassifierTrainer
import smile.data.Attribute
import java.io.InputStream
import java.io.OutputStream
import java.util.ArrayList
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

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
open class SmileClassifier : TargetSupervisedLearningTransform<String>, MultiFileMixedTransform {
    /**
     * This is the trained model. It won't be set until the training phase of trainTransform is complete.
     */
    @JsonIgnore
    lateinit var trainedModel: Classifier<DoubleArray>

    // The facotry used for getting a trainer. Will be null when we deserialize a trained model
    private val trainerFactory: ((Array<Attribute>) -> ClassifierTrainer<DoubleArray>)?

    val predictionColName: String

    /**
     * Allows us to map from the predicted values, which are integers, back to Strings. Will not be available until
     * after the training phase has completed.
     */
    lateinit var intToTarget: Map<Int, String>

    constructor(targetId: ColumnId<String>,
            trainerFactory: (Array<Attribute>) -> ClassifierTrainer<DoubleArray>,
            predictionColName: String = "predicted") : super(targetId) {
        this.trainerFactory = trainerFactory
        this.predictionColName = predictionColName
    }

    @JsonCreator
    constructor(
            @JacksonInject(MultiFileMixedFormat.INJECTED_BINARY_DATA) trainModelInputStream: InputStream,
            @JsonProperty("predictionColName") predictionColName: String) : super(null) {
        trainerFactory = null
        this.predictionColName = predictionColName

        @Suppress("UNCHECKED_CAST")
        trainedModel = xstream.fromXML(trainModelInputStream) as Classifier<DoubleArray>
    }

    companion object {
        private val log = LoggerFactory.getLogger(SmileClassifier::class.java)
        private val xstream = XStream()
    }

    override fun transform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        val dataAndAttrs = DataConversion.fromDataSet(dataSet)
        return CompletableFuture.completedFuture(transformConvertedData(dataAndAttrs.data))
    }

    override fun trainTransform(dataSet: DataSet, target: Column<String?>, exec: ExecutorService)
            : CompletableFuture<DataSet> {
        log.info("Converting data set to smile format")
        val dataAndAttrs = DataConversion.fromDataSet(dataSet)
        log.info("Constructing smile classifier instance")
        val trainer = trainerFactory!!.invoke(dataAndAttrs.attributes)
        log.info("Converting target data")
        val targetInfo = DataConversion.toCategoricalTarget(target)
        intToTarget = targetInfo.intToStringMapping
        log.info("Training smile classifier of type {}", trainer.javaClass)
        trainedModel = trainer.train(dataAndAttrs.data, targetInfo.target)
        log.info("Training complete. Calling transform.")
        return CompletableFuture.completedFuture(transformConvertedData(dataAndAttrs.data))
    }

    private fun transformConvertedData(data: Array<DoubleArray>): DataSet {
        val predictions = ArrayList<String>(data.size)
        for (row in data) {
            val intPred = trainedModel.predict(row)
            val prediction = intToTarget[intPred] ?:
                    throw IllegalStateException("$intPred was predicted but isn't a known target value")
            predictions.add(prediction)
        }
        val resultCol = ColumnId.create<String>(predictionColName)
        val resultMeta = CategoricalFeatureMetaData(false, intToTarget.values.toSet())
        return DataSet.build {
            addColumn(resultCol, predictions, resultMeta)
        }
    }

    override fun serializeBinaryData(output: OutputStream) {
        xstream.toXML(trainedModel, output)
    }
}
