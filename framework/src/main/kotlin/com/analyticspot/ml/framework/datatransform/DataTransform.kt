package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.observation.Observation

/**
 * A class for transformations that take a single input and produce a single output. These transforms do no learn from
 * training data; they simply make a transformation. See [LearningTransform] or similar for transformation that need to
 * be trained before they can be used.
 */
interface DataTransform {
    /**
     * Describes the outputs produced by this transformation.
     */
    val description: TransformDescription

    fun transform(observation: Observation): Observation
}
