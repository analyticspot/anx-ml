package com.analyticspot.ml.framework.metadata

import com.analyticspot.ml.framework.dataset.Column

/**
 * A [CategoricalFeatureId] is a metadata that takes a value from a set of possible values. The values are not in any way
 * sortable or comparable. For example, consider something like "car color" which could take on values like "red",
 * "blue", "green", etc. Note that we can't compare "red" to "green" in any meaningful way so this is a categorical
 * metadata.
 */
class CategoricalFeatureMetaData(maybeMissing: Boolean, val possibleValues: Set<String>)
    : MaybeMissingMetaData(maybeMissing) {
    companion object {
        /**
         * Convenience constructor that takes a `Column<String>` and returns a [CategoricalFeatureId] whose possible
         * values come from the values observed in the input. Note that if the provided `name` is the same as the
         * `name` for the current [ColumnId] the retured [CategoricalFeatureId] can be used in place of the `ColumnId`
         * without any changes to the [DataSet].
         *
         * @param input the data to use to find the set of possible values
         * @param name the name for the returned [CategoricalFeatureId]
         * @param maybeMissing if null, the returned [CategoricalFeatureId] will have [maybeMissing] as true if and
         *     only if the `input` contained one or more `null` values. If non-null the given value will be used (and
         *     checked for accuracy).
         */
        fun fromStringColumn(input: Column<String?>, maybeMissing: Boolean? = null): CategoricalFeatureMetaData {
            val possibleValues = mutableSetOf<String>()
            var observedMissing = false
            input.forEach {
                if (it != null) {
                    possibleValues.add(it)
                } else {
                    observedMissing = true
                }
            }
            require(maybeMissing == null || maybeMissing || (!maybeMissing && !observedMissing)) {
                "You passed false for maybeMissing but the input contains missing values."
            }
            return CategoricalFeatureMetaData(maybeMissing ?: observedMissing, possibleValues)
        }
    }
}
