/*
 * Copyright (C) 2017 Analytic Spot.
 * 
 * This file is part of the ANX ML library.
 * 
 * The ANX ML library is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser
 * General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 * 
 * The ANX ML library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
 * for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License along with the ANX ML library.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.utils.isAssignableFrom
import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonProperty
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 * Takes in a source [DataSet] and returns a new [DataSet] that contains a subset of the columns in the source. There
 * is also an option to rename the columns that are retained. If one of the retained columns has metadata that is
 * retained as well.
 */
class ColumnSubsetTransform : SingleDataTransform {
    /**
     * The keys in this map are the columns that will be retained. The values are new names for the columns. If the
     * column isn't going to be renamed then the value is the same as the key.
     */
    val keepMap: Map<String, String>

    @JsonCreator
    private constructor(
            @JsonProperty("keepMap") keepMap: Map<String, String>) {
        this.keepMap = keepMap
    }

    private constructor(builder: Builder) : this(builder.keepMap)

    companion object {
        fun builder(): Builder = Builder()

        fun build(init: Builder.() -> Unit): ColumnSubsetTransform {
            return with(builder()) {
                init()
                build()
            }
        }
    }

    override fun transform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        val bldr = DataSet.builder()
        keepMap.forEach { entry ->
            val md = dataSet.metaData[entry.key]
            val srcCol = dataSet.columnIdWithNameUntyped(entry.key)
            val newCol = ColumnId(entry.value, srcCol.clazz)
            @Suppress("UNCHECKED_CAST")
            bldr.addColumn(newCol as ColumnId<Any>, dataSet.column(srcCol), md)
        }
        return CompletableFuture.completedFuture(bldr.build())
    }

    class Builder {
        // Maps from the id of a column to keep to a new id for that column.
        internal val keepMap = mutableMapOf<String, String>()

        /**
         * The result should contain the given column.
         */
        fun keep(id: ColumnId<*>): Builder {
            return keepAndRename(id, id.name)
        }

        /**
         * The same as the other overload but allows you to specify column names instead of [ColumnId] instances.
         */
        fun keep(name: String): Builder {
            return keepAndRename(name, name)
        }

        /**
         * The result should contain `srcCol` but in the output that data will have name `newName`.
         */
        fun keepAndRename(srcCol: ColumnId<*>, newName: String): Builder {
            return keepAndRename(srcCol.name, newName)
        }

        /**
         * The result should contain `srcCol` but in the output that data will have name `newName`.
         */
        fun keepAndRename(srcCol: ColumnId<*>, newCol: ColumnId<*>): Builder {
            require(newCol.clazz isAssignableFrom srcCol.clazz)
            return keepAndRename(srcCol.name, newCol.name)
        }

        /**
         * The same as the other overload but allows you to specify column names instead of [ColumnId] instances.
         */
        fun keepAndRename(srcCol: String, newName: String): Builder {
            require(srcCol !in keepMap) {
                "Column $srcCol already specified"
            }
            keepMap[srcCol] = newName
            return this
        }

        fun build(): ColumnSubsetTransform {
            return ColumnSubsetTransform(this)
        }
    }
}
