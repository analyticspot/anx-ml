package com.analyticspot.ml.framework.feature

import com.analyticspot.ml.framework.dataset.Column
import com.analyticspot.ml.framework.description.ColumnId

/**
 * A [CategoricalFeatureId] is a feature that takes a value from a set of possible values. The values are not in any way
 * sortable or comparable. For example, consider something like "car color" which could take on values like "red",
 * "blue", "green", etc. Note that we can't compare "red" to "green" in any meaningful way so this is a categorical
 * feature.
 */
class CategoricalFeatureId(name: String, maybeMissing: Boolean, val possibleValues: Set<String>)
    : FeatureId<String>(name, String::class, maybeMissing) {
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
         *     only if the `input` contained one or more `null` values. If non-null the given value will be used.
         */
        fun fromStringColumn(input: Column<String?>, name: String, maybeMissing: Boolean? = null): CategoricalFeatureId {
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
            return CategoricalFeatureId(name, maybeMissing ?: observedMissing, possibleValues)
        }

        /**
         * Converts a [ColumnId] of type `String` into the corresponding [CategoricalFeatureId].
         */
        fun fromColumnId(src: ColumnId<String>, maybeMissing: Boolean,
                possibleValues: Set<String>): CategoricalFeatureId {
            return CategoricalFeatureId(src.name, maybeMissing, possibleValues)
        }
    }
}
