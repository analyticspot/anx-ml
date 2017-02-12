package com.analyticspot.ml.bridges.smile

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.metadata.CategoricalFeatureMetaData
// Lint disable as this is used but there's a ktlint bug.
import com.analyticspot.ml.utils.isAssignableFrom // ktlint-disable no-unused-imports
import smile.data.Attribute
import smile.data.NominalAttribute
import smile.data.NumericAttribute

/**
 * Smile has an `AttributeDataSet` but they don't ever pass that directly to classifiers, clusterers, etc. Instead
 * they pass the `Attribute` array in one place and then pass the features as a `double[][]` and the target as a
 * `double[]`. Thus this object contains routines for converting our `Column<T>` instances into Smile `Attribute`. It is
 * assumed that columns of type `Int`, `Long`, or `Double` are numeric, columns of type `String` are categorical, and
 * columns of type `Boolean` are boolean, which, in Smile, is treated as just a nominal attribute with only two possible
 * values.
 */
object AttributeConversion {
    /**
     * Converts all the [FeatureId] in the [DataSet] into a corresponding array of smile `Attribute`. The
     * returned array is guaranteed to be sorted lexicographically by metadata/attribute name.
     *
     * This requires that all the [ColumnId] in the [DataSet] are [FeatureId] instances. If not, this will throw an
     * illegal argument exception.
     */
    @Suppress("UNCHECKED_CAST")
    fun toSmileAttributes(inData: DataSet): Array<Attribute> {
        return Array<Attribute>(inData.numColumns) { colIdx ->
            val colId = inData.columnIds[colIdx]
            if (Number::class isAssignableFrom colId.clazz) {
                toAttribute(colId as ColumnId<in Number>)
            } else if (String::class isAssignableFrom colId.clazz) {
                val md = inData.metaData[colId.name]
                if (md != null && md is CategoricalFeatureMetaData) {
                    toAttribute(colId as ColumnId<in String>, md)
                } else {
                    throw IllegalArgumentException("Found string column ${colId.name} but no metadata")
                }
            } else if (Boolean::class isAssignableFrom colId.clazz) {
                toAttribute(colId as ColumnId<in Boolean>)
            } else {
                throw IllegalArgumentException("No known conversion for columns of type ${colId.clazz}")
            }
        }
    }

    /**
     * Converts a `String` column with meta data into a `NominalAttribute`.
     */
    fun toAttribute(catId: ColumnId<in String>, metaData: CategoricalFeatureMetaData): NominalAttribute {
        return NominalAttribute(catId.name, metaData.possibleValues.toTypedArray())
    }

    /**
     * Converts one of our [NumericalFeatureId] instances into a smile `NumericAttribute`.
     */
    fun toAttribute(numId: ColumnId<in Number>): NumericAttribute {
        return NumericAttribute(numId.name)
    }

    /**
     * Converts one of our [BooleanFeatureId] instance to a smile `NominalAttribute`. Smile doesn't have a special type
     * for boolean/binary variables; they just treat them as a nominal with only two possible values
     */
    fun toAttribute(boolId: ColumnId<in Boolean>): NominalAttribute {
        return NominalAttribute(boolId.name, arrayOf("false", "true"))
    }
}

