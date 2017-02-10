package com.analyticspot.ml.bridges.smile

import com.analyticspot.ml.framework.dataset.FeatureDataSet
import com.analyticspot.ml.framework.feature.CategoricalFeatureId
import com.analyticspot.ml.framework.feature.NumericalFeatureId
import smile.data.Attribute
import smile.data.NominalAttribute
import smile.data.NumericAttribute

/**
 * Smile has an `AttributeDataSet` but they don't ever pass that directly to classifiers, clusterers, etc. Instead
 * they pass the `Attribute` array in one place and then pass the features as a `double[][]` and the target as a
 * `double[]`. Thus this object contains routines for converting our `FeatureDataSet` instances into arrays of
 * attributes, arrays of doubles, etc.
 */
object AttributeConversion {
    /**
     * Converts all the [FeatureId] in the [FeatureDataSet] into a corresponding array of smile `Attribute`. The
     * returned array is guaranteed to be sorted lexicographically by feature/attribute name.
     */
    fun toSmileAttributes(inData: FeatureDataSet): Array<Attribute> {
        return Array<Attribute>(inData.numColumns) { colIdx ->
            val colId = inData.columnIds[colIdx]
            when (colId) {
                is CategoricalFeatureId -> toAttribute(colId)
                is NumericalFeatureId -> toAttribute(colId)
                else -> throw IllegalArgumentException("Unknown feature type ${colId.javaClass}")
            }
        }
    }

    /**
     * Converts one of our [CategoricalFeatureId] instances to a smile `NominalAttribute` instance.
     */
    fun toAttribute(catId: CategoricalFeatureId): NominalAttribute {
        return NominalAttribute(catId.name, catId.possibleValues.toTypedArray())
    }

    /**
     * Converts one of our [NumericalFeatureId] instances into a smile `NumericAttribute`.
     */
    fun toAttribute(numId: NumericalFeatureId): NumericAttribute {
        return NumericAttribute(numId.name)
    }
}

