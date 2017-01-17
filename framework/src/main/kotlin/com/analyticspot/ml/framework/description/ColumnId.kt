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

import com.analyticspot.ml.framework.serialization.JsonMapper
import kotlin.reflect.KClass

/**
 * A value id allows the user to obtain data from a [DataSet] in a type safe way.
 *
 * By convention groups of related values are named with a prefix, a separator of `-` and a suffix. Thus, if you are
 * naming just a single feature/data item refrain from using the `-` character.
 */
open class ColumnId<DataT : Any>(val name: String, val clazz: KClass<DataT>) : Comparable<ColumnId<*>> {
    /**
     * Alternative constructor taking Java classes.
     */
    constructor(name: String, clazz: Class<DataT>) : this(name, clazz.kotlin)

    companion object {
        inline fun <reified T : Any> create(name: String) = ColumnId<T>(name, T::class)
        /**
         * The character used to separate the prefix and the tokens in a [ValueTokenGroup].
         */
        const val GROUP_SEPARATOR = "-"
    }

    final override fun compareTo(other: ColumnId<*>): Int {
        if (name == other.name) {
            check(clazz.javaObjectType == other.clazz.javaObjectType) {
                "Somehow there's two ColumnId instances named $name with different types: " +
                    "[$clazz] and [${other.clazz}]"
            }
            return 0
        } else {
            return name.compareTo(other.name)
        }
    }

    final override fun equals(other: Any?): Boolean {
        if (this === other) return true

        if (other is ColumnId<*>) {
            return compareTo(other) == 0
        } else {
            return false
        }
    }

    final override fun hashCode(): Int {
        return name.hashCode()
    }

    override fun toString(): String {
        return JsonMapper.mapper.writeValueAsString(this)
    }
}
