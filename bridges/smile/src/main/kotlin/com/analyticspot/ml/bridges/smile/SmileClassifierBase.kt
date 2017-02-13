package com.analyticspot.ml.bridges.smile

import com.analyticspot.ml.framework.dataset.Column
import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.datatransform.TargetSupervisedLearningTransform
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.serialization.MultiFileMixedTransform
import com.fasterxml.jackson.annotation.JsonIgnore
import com.thoughtworks.xstream.XStream
import org.slf4j.LoggerFactory
import smile.classification.Classifier
import smile.classification.ClassifierTrainer
import smile.data.Attribute
import java.io.InputStream
import java.io.OutputStream
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 * Base class for wrapping smile classifiers
 */
abstract class SmileClassifierBase<ClassifierT : Classifier<DoubleArray>>
    : TargetSupervisedLearningTransform<String>, MultiFileMixedTransform {
    /**
     * This is the trained model. It won't be set until the training phase of trainTransform is complete.
     */
    @JsonIgnore
    lateinit var trainedModel: ClassifierT

    // The facotry used for getting a trainer. Will be null when we deserialize a trained model
    private val trainerFactory: ((Array<Attribute>) -> ClassifierTrainer<DoubleArray>)?

    val predictionCol: ColumnId<String>

    /**
     * Allows us to map from the predicted values, which are integers, back to Strings. Will not be available until
     * after the training phase has completed.
     */
    lateinit var intToTarget: Map<Int, String>

    constructor(targetId: ColumnId<String>,
            trainerFactory: (Array<Attribute>) -> ClassifierTrainer<DoubleArray>,
            predictionCol: ColumnId<String> = ColumnId.create<String>("predicted")) : super(targetId) {
        this.trainerFactory = trainerFactory
        this.predictionCol = predictionCol
    }

    // This constructor is for Jackson deserialization.
    constructor(trainModelInputStream: InputStream,
            predictionCol: ColumnId<String>) : super(null) {
        trainerFactory = null
        this.predictionCol = predictionCol

        @Suppress("UNCHECKED_CAST")
        trainedModel = xstream.fromXML(trainModelInputStream) as ClassifierT
    }

    companion object {
        private val log = LoggerFactory.getLogger(SmileClassifierBase::class.java)
        private val xstream = XStream()
    }

    override final fun transform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        val dataAndAttrs = DataConversion.fromDataSet(dataSet)
        return CompletableFuture.completedFuture(transformConvertedData(dataAndAttrs.data))
    }

    override final fun trainTransform(dataSet: DataSet, target: Column<String?>, exec: ExecutorService)
            : CompletableFuture<DataSet> {
        log.info("Converting data set to smile format")
        val dataAndAttrs = DataConversion.fromDataSet(dataSet)
        log.info("Constructing smile classifier instance")
        val trainer = trainerFactory!!.invoke(dataAndAttrs.attributes)
        log.info("Converting target data")
        val targetInfo = DataConversion.toCategoricalTarget(target)
        intToTarget = targetInfo.intToStringMapping
        log.info("Training smile classifier of type {}", trainer.javaClass)
        // No way to avoid this - the Smile type hierarchy has only 1 type of trainer but it will return different
        // subclasses of Classifier
        @Suppress("UNCHECKED_CAST")
        trainedModel = trainer.train(dataAndAttrs.data, targetInfo.target) as ClassifierT
        log.info("Training complete. Calling transform.")
        return CompletableFuture.completedFuture(transformConvertedData(dataAndAttrs.data))
    }

    // Called by the base class with data for which we want predictions. This is abstract as plain-old classifiers can
    // only provide us with a prediction while soft classifiers can provide a prediction column plus one column per
    // target value for the posterior probability for that column.
    abstract protected fun transformConvertedData(data: Array<DoubleArray>): DataSet

    override fun serializeBinaryData(output: OutputStream) {
        xstream.toXML(trainedModel, output)
    }
}
