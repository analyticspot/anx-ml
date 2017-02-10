package com.analyticspot.ml.framework.feature

import com.analyticspot.ml.framework.description.ColumnId
import kotlin.reflect.KClass

/**
 * A [FeatureId] is just a [ColumnId] plus some additional metadata. This is the base class. There are subclasses for the
 * common machine learning feature types like [CategoricalFeatureId], [NumericalFeature], etc.
 *
 * This class is really just a marker that the subclass is some kind of feature and hence usable by our ML algorithms.
 *
 * @param name the name as per [ColumnId.name]
 * @param clazz the type of the data as per [ColumnId.clazz]
 * @param maybeMissing if true, this feature may contain missing values. If false we can assume that all rows have
 *     non-null values.
 */
open class FeatureId<DataT : Any>(name: String, clazz: KClass<DataT>, val maybeMissing: Boolean)
    : ColumnId<DataT>(name, clazz)
