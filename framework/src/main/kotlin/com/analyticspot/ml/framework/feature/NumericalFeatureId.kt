package com.analyticspot.ml.framework.feature

import com.analyticspot.ml.framework.description.ColumnId

/**
 * A feature that contains numeric data. For simplicity and compatability with most ML libraries we require that all
 * numeric features are of type `Double`.
 */
class NumericalFeatureId(name: String, maybeMissing: Boolean) : FeatureId<Double>(name, Double::class, maybeMissing) {
    companion object {
        fun fromColumnId(colId: ColumnId<Double>, maybeMissing: Boolean): NumericalFeatureId {
            return NumericalFeatureId(colId.name, maybeMissing)
        }
    }
}
