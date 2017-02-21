package com.analyticspot.ml.framework.metadata

import com.analyticspot.ml.framework.dataset.Column

/**
 * A [CategoricalFeatureMetaData] is a metadata that takes a value from a set of possible values. The values are not in
 * any way sortable or comparable. For example, consider something like "car color" which could take on values like
 * "red", "blue", "green", etc. Note that we can't compare "red" to "green" in any meaningful way so this is a
 * categorical metadata.
 */
class CategoricalFeatureMetaData(maybeMissing: Boolean, val possibleValues: Set<String>)
    : MaybeMissingMetaData(maybeMissing) {
    companion object {
        /**
         * Convenience constructor that takes a `Column<String>` and returns a [CategoricalFeatureMetaData] whose
         * possible values come from the values observed in the input.
         *
         * @param input the data to use to find the set of possible values
         * @param maybeMissing if null, the returned [CategoricalFeatureMetaData] will have [maybeMissing] as true if
         *     and only if the `input` contained one or more `null` values. If non-null the given value will be used
         *     (and checked for accuracy).
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

    /**
     * Given a [Column] of `String` values this returns a `Boolean` indicating if the column conforms to the metadata.
     * Specifically, the column should contain only values in [possibleValues]. To replace unknown values with `null` you
     * can call [makeColumnConformToMetaData] instead.
     */
    fun doesColumnConformToMetaData(column: Column<String?>): Boolean {
        return column.all {
            it == null || it in possibleValues
        }
    }

    /**
     * Replaces any values in the input column that are not in [possibleValues] with `null`. Note that if [maybeMissing]
     * is `false` the resulting column will no longer be compatible with this metadata. To get compatible metadata call
     * [copyWithMissingAllowed].
     */
    fun makeColumnConformToMetaData(column: Column<String?>): Column<String?> {
        return column.mapToColumn {
            if (it == null || it in possibleValues) {
                it
            } else {
                null
            }
        }
    }

    /**
     * Returns a new [CategoricalFeatureMetaData] instance that's identical to this one except that [maybeMissing] is
     * `true`.
     */
    fun copyWithMissingAllowed(): CategoricalFeatureMetaData {
        if (maybeMissing) {
            return this
        } else {
            return CategoricalFeatureMetaData(true, possibleValues)
        }
    }
}
