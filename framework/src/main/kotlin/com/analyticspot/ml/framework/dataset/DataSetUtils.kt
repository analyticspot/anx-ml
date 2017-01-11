package com.analyticspot.ml.framework.dataset

import com.analyticspot.ml.framework.observation.Observation
import kotlinx.support.jdk8.collections.spliterator
import java.util.stream.Stream
import java.util.stream.StreamSupport

/**
 * A collection of utility methods for creating and manipulating [DataSet] objects.
 */

/**
 * Create a [DataSet] from a list of Observation.
 */
fun createDataSet(vararg observations: Observation): DataSet = ArrayDataSet(observations)

fun toStream(dataSet: DataSet): Stream<Observation> {
    return StreamSupport.stream(dataSet.spliterator(), false)
}
