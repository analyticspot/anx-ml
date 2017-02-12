package com.analyticspot.ml.bridges.smile

import com.analyticspot.ml.framework.dataset.Column
import com.analyticspot.ml.framework.dataset.DataSet
// Lint disable as this is used but there's a ktlint bug.
import com.analyticspot.ml.utils.isAssignableFrom // ktlint-disable no-unused-imports
import smile.data.NominalAttribute

/**
 * Code to convert our `FeatureDataSet` instances into arrays of doubles as understood by Smile. Smile understands only
 * doubles so things like [CategoricalFeatureId] must be converted into doubles in a way that smile understands. Also,
 * smile encodes missing values as NaN so this conversion must happen as well.
 */
object DataConversion {
    /**
     * Converts all the data in the [DataSet] to [DataAndAttrs]. This requires that all the [ColumnId] instances in
     * `dataSet` are instances of [FeatureId]; if not, this will throw an `IllegalArgumentException`.
     */
    fun fromDataSet(dataSet: DataSet): DataAndAttrs {
        // First get all the attributes
        val attrs = AttributeConversion.toSmileAttributes(dataSet)

        assert(attrs.map { it.name } == dataSet.columnIds.map { it.name })

        // Now convert all the rows of data
        val data = Array<DoubleArray>(dataSet.numRows) { rowIdx ->
            DoubleArray(dataSet.numColumns) { colIdx ->
                val colId = dataSet.columnIds[colIdx]
                if (Number::class isAssignableFrom colId.clazz) {
                    numberToDouble(dataSet.value(rowIdx, colId) as Number?)
                } else if (String::class isAssignableFrom colId.clazz || Boolean::class isAssignableFrom colId.clazz) {
                    categoricalOrBooleanToDouble(dataSet.value(rowIdx, colId)?.toString(),
                            attrs[colIdx] as NominalAttribute)
                } else {
                    throw IllegalArgumentException("Unknown column type ${colId.javaClass}")
                }
            }
        }

        return DataAndAttrs(data, attrs)
    }

    /**
     * Converts any type of number (Int, Long, etc.) to a double. If `input` is null returns `NaN`.
     */
    fun numberToDouble(input: Number?): Double {
        if (input == null) {
            return Double.NaN
        } else {
            return input.toDouble()
        }
    }

    /**
     * Converts a single categorical or boolean value (Smile treats both as Nominal) to the correct Double.
     */
    fun categoricalOrBooleanToDouble(value: String?, attr: NominalAttribute): Double {
        if (value == null) {
            return Double.NaN
        } else {
            // Let smile do the conversion from String to double so it's consistent in how it's done.
            return attr.valueOf(value)
        }
    }

    /**
     * Converts a [Column] of type String to a `IntArray` for use as a target with smile Classifier instances. Also
     * returns the mapping from string values to integers so we can reverse the conversion. Since this is to be a target
     * variable this will throw if any of the values are missing.
     */
    fun toCategoricalTarget(col: Column<String?>): CategoricalTarget {
        val mapping = mutableMapOf<String, Int>()
        val targetData = IntArray(col.size) { idx ->
            val item = col[idx]
            if (item == null) {
                throw IllegalArgumentException("Target data cannot contain any missing values.")
            } else {
                if (item in mapping) {
                    mapping[item]!!
                } else {
                    val newVal = mapping.size
                    mapping[item] = newVal
                    newVal
                }
            }
        }
        return CategoricalTarget(targetData, mapping)
    }
}
