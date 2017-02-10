package com.analyticspot.ml.bridges.smile

import com.analyticspot.ml.framework.dataset.Column
import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.datatransform.TargetSupervisedLearningTransform
import com.analyticspot.ml.framework.description.TransformDescription
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 * Created by oliver on 2/10/17.
 */
class SmileClassifier : TargetSupervisedLearningTransform<Int>() {
    override val description: TransformDescription
        get() = throw UnsupportedOperationException()

    override fun transform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        throw UnsupportedOperationException("not implemented") //To change body of created functions use File | Settings | File Templates.
    }


    override fun trainTransform(dataSet: DataSet, target: Column<Int?>, exec: ExecutorService): CompletableFuture<DataSet> {
        throw UnsupportedOperationException("not implemented") //To change body of created functions use File | Settings | File Templates.
    }
}
