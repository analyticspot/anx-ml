package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.observation.Observation

/**
 * A class for transformations that take one input DataSet and produce one output DataSet. These transforms do no learn
 * from training data; they simply make a transformation. See [LearningTransform] or similar for transformation that
 * need to be trained before they can be used.
 */
interface DataTransform {
    /**
     * Describes the outputs produced by this transformation.
     */
    val description: TransformDescription

    fun transform(dataSet: DataSet): DataSet
}

