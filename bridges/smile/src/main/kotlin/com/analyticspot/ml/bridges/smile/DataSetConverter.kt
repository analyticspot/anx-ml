package com.analyticspot.ml.bridges.smile

import com.analyticspot.ml.framework.dataset.FeatureDataSet
import com.analyticspot.ml.framework.feature.CategoricalFeatureId
import smile.data.Attribute
import smile.data.NominalAttribute

/**
 * Smile has an `AttributeDataSet` but they don't ever pass that directly to classifiers, clusterers, etc. Instead
 * they pass the `Attribute` array in one place and then pass the features as a `double[][]` and the target as a
 * `double[]`. Thus this object contains routines for converting our `FeatureDataSet` instances into arrays of
 * attributes, arrays of doubles, etc.
 */
object AttributeConversion {
    /**
     * Converts all the [FeatureId] in the [FeatureDataSet] into a corresponding array of smile `Attribute`.
     */
//    fun toSmileAttributes(inData: FeatureDataSet): Array<Attribute> {
//
//        for (featureId in inData.columnIds) {
//
//        }
//    }

    fun toAttribute(catId: CategoricalFeatureId): NominalAttribute {
        return NominalAttribute(catId.name, catId.possibleValues.toTypedArray())
    }

}




