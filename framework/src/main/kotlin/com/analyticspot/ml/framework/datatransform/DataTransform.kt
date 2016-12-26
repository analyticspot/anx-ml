package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.observation.Observation

/**
 *
 */
interface DataTransform {
    val description: TransformDescription

    fun transform(observation: Observation): Observation
}
