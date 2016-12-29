package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.dataset.IterableDataSet
import com.analyticspot.ml.framework.datatransform.LearningTransform
import com.analyticspot.ml.framework.datatransform.TransformDescription
import com.analyticspot.ml.framework.description.ValueId
import com.analyticspot.ml.framework.description.ValueToken
import com.analyticspot.ml.framework.observation.SingleValueObservation
import org.slf4j.LoggerFactory
import java.util.concurrent.CompletableFuture

class LearnMinTransform(private val srcToken: ValueToken<Int>, resultId: ValueId<Int>) : LearningTransform {
    private var minValue: Int = Int.MAX_VALUE
    private val resultToken = ValueToken(resultId)
    override val description: TransformDescription
        get() = TransformDescription(listOf(resultToken))

    companion object {
        private val log = LoggerFactory.getLogger(LearnMinTransform::class.java)
    }

    override fun trainTransform(dataSet: DataSet): CompletableFuture<DataSet> {
        log.debug("{} is training", this.javaClass)
        val dsMin = dataSet.asSequence().map { it.value(srcToken) }.min()
        minValue = dsMin ?: minValue
        return transform(dataSet)
    }

    override fun transform(dataSet: DataSet): CompletableFuture<DataSet> {
        val resultData = dataSet.asSequence().map {
            SingleValueObservation.create(minValue)
        }.toList()
        return CompletableFuture.completedFuture(IterableDataSet(resultData))
    }
}
