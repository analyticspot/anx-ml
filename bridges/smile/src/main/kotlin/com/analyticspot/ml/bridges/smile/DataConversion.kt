package com.analyticspot.ml.bridges.smile

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.feature.CategoricalFeatureId
import com.analyticspot.ml.framework.feature.NumericalFeatureId
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
                when (colId) {
                    is CategoricalFeatureId -> categoricalToDouble(dataSet.value(rowIdx, colId),
                            attrs[colIdx] as NominalAttribute)
                    is NumericalFeatureId -> dataSet.value(rowIdx, colId) ?: Double.NaN
                    else -> throw IllegalArgumentException("Unknown column type ${colId.javaClass}")
                }
            }
        }

        return DataAndAttrs(data, attrs)
    }

    fun categoricalToDouble(value: String?, attr: NominalAttribute): Double {
        if (value == null) {
            return Double.NaN
        } else {
            // Let smile do the conversion from String to double so it's consistent in how it's done.
            return attr.valueOf(value)
        }
    }
}
