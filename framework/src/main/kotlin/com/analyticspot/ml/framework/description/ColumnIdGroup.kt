/*
 * Copyright (C) 2017 Analytic Spot.
 * 
 * This file is part of the ANX ML library.
 * 
 * The ANX ML library is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser
 * General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 * 
 * Foobar is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License along with the ANX ML libarary.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

package com.analyticspot.ml.framework.description

import kotlin.reflect.KClass

/**
 * A single id that refers to a set of [ColumnId] all with the same type. This is used by some algorithms which produce
 * an unknown number of values. For example, a "bag of words" transform will produce one `Int` value for each unique
 * word in the training corpus: until it's been trained we can't know many words there will be. Still, we often want to
 * tell consuming [DataTransformer] which values to consume from a [DataSet] so we need a way to refer to a group of
 * columns. This is done via a [ColumnIdGroup]: it refers to an unknown number of [ColumnId] that all have the same type
 * and whose names all have the same prefix.
 */
open class ColumnIdGroup<T : Any>(val prefix: String, val clazz: KClass<T>) : Comparable<ColumnIdGroup<T>> {
    init {
        check(!prefix.contains(ColumnId.GROUP_SEPARATOR)) {
            "ValueIdGroup prefixes should not contain the character ${ColumnId.GROUP_SEPARATOR}"
        }
    }

    companion object {
        inline fun <reified T : Any> create(prefix: String): ColumnIdGroup<T> {
            return ColumnIdGroup(prefix, T::class)
        }
    }

    /**
     * Generate a column-id for the given suffix.
     */
    fun generateId(suffix: String): ColumnId<T> = ColumnId(prefix + ColumnId.GROUP_SEPARATOR + suffix, clazz)

    override fun compareTo(other: ColumnIdGroup<T>): Int {
        return prefix.compareTo(other.prefix)
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other?.javaClass != javaClass) return false

        other as ColumnIdGroup<*>

        return prefix == other.prefix
    }

    override fun hashCode(): Int {
        return prefix.hashCode()
    }
}

