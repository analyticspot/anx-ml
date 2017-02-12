package com.analyticspot.ml.bridges.smile

import com.analyticspot.ml.framework.dataset.Column
import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.datatransform.SupervisedLearningTransform
import com.analyticspot.ml.framework.datatransform.TargetSupervisedLearningTransform
import com.analyticspot.ml.framework.feature.CategoricalFeatureId
import org.slf4j.LoggerFactory
import smile.classification.Classifier
import smile.classification.ClassifierTrainer
import smile.data.Attribute
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 * Created by oliver on 2/10/17.
 */
class SmileClassifier(targetId: CategoricalFeatureId,
        private val trainerFactory: (Array<Attribute>) -> ClassifierTrainer<DoubleArray>)
    : TargetSupervisedLearningTransform<String>(targetId) {
    /**
     * This is the trained model. It won't be set until the training phase of trainTransform is complete.
     */
    lateinit var trainedModel: Classifier<DoubleArray>

    /**
     * Allows us to map from the predicted values, which are integers, back to Strings. Will not be available until
     * after the training phase has completed.
     */
    lateinit var intToTarget: Map<Int, String>


    companion object {
        private val log = LoggerFactory.getLogger(SmileClassifier::class.java)
    }

    override fun transform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        throw UnsupportedOperationException("not implemented") //To change body of created functions use File | Settings | File Templates.
    }


    override fun trainTransform(dataSet: DataSet, target: Column<String?>, exec: ExecutorService)
            : CompletableFuture<DataSet> {
        log.info("Converting data set to smile format")
        val dataAndAttrs = DataConversion.fromDataSet(dataSet)
        log.info("Constructing smile classifier instance")
        val trainer = trainerFactory.invoke(dataAndAttrs.attributes)
        log.info("Converting target data")
        val targetInfo = DataConversion.toCategoricalTarget(target)
        intToTarget = targetInfo.intToStringMapping
        log.info("Training smile classifier of type {}", trainer.javaClass)
        trainedModel = trainer.train(dataAndAttrs.data, targetInfo.target)
        log.info("Training complete. Calling transform.")
        return CompletableFuture.completedFuture(transformConvertedData(dataAndAttrs.data))
    }

    private fun transformConvertedData(data: Array<DoubleArray>): DataSet {

    }
}
