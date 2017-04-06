package com.analyticspot.ml.framework.utils

import com.analyticspot.ml.framework.dataset.Column

/**
 * Various utilities for working with data.
 */
object DataUtils {
    /**
     * Converts a [Column] of type `String` into a [Column] of type `Int` by mapping each unique string to a different
     * integer in the range [0, n] where `n` is the total number of unique values in the column. Returns the new column
     * and a map from integer to string that allows you do reverse the transform.
     */
    fun encodeCategorical(data: Column<String?>): Pair<Column<Int>, Map<Int, String>> {
        val strToInt = mutableMapOf<String, Int>()
        val newCol: Column<Int> = data.mapToColumn {
            var intEncoding = strToInt[it]
            if (intEncoding == null) {
                intEncoding = strToInt.size
                strToInt[it!!] = intEncoding
            }
            // The null assert really is unnecessary but the compiler complains that it's required if it's missing and
            // that it's unnecessary if it's present...
            @Suppress("UNNECESSARY_NOT_NULL_ASSERTION")
            intEncoding!!
        }

        return Pair(newCol, strToInt.asSequence().associate { it.value to it.key })
    }

    /**
     * Reversed [encodeCategorical].
     *
     * @param decoderRing the `Map` returned by [encodeCategorical].
     * @param toDecode the [Column] returned by [encodeCategorical].
     */
    fun decodeCategorical(decoderRing: Map<Int, String>, toDecode: Column<Int>): Column<String> {
        return toDecode.mapToColumn {
            decoderRing[it]!!
        }
    }
}
