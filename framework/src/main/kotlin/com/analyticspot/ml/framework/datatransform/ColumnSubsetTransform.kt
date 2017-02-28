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
 * You should have received a copy of the GNU Lesser General Public License along with the ANX ML libarary.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonIgnore
import com.fasterxml.jackson.annotation.JsonProperty
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 * Takes in a source [DataSet] and returns a new [DataSet] that contains a subset of the columns in the source. There
 * is also an option to rename the columns that are retained. If one of the retained columns has metadata that is
 * retained as well.
 */
class ColumnSubsetTransform : SingleDataTransform {
    @JsonIgnore
    val keepMap: Map<ColumnId<*>, ColumnId<*>>

    @JsonCreator
    private constructor(
            @JsonProperty("keepMap") keepMap: Map<ColumnId<*>, ColumnId<*>>) {
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
            val md = dataSet.metaData[entry.key.name]
            @Suppress("UNCHECKED_CAST")
            bldr.addColumn(entry.value as ColumnId<Any>, dataSet.column(entry.key), md)
        }
        return CompletableFuture.completedFuture(bldr.build())
    }

    class Builder {
        // Maps from the id of a column to keep to a new id for that column.
        internal val keepMap = mutableMapOf<ColumnId<*>, ColumnId<*>>()

        /**
         * The result should contain the given column.
         */
        fun keep(id: ColumnId<*>): Builder {
            require(id !in keepMap) {
                "Column $id already specified"
            }
            keepMap[id] = id
            return this
        }

        /**
         * The result should contain `srcCol` but in the output that data will have name `newName`.
         */
        fun keepAndRename(srcCol: ColumnId<*>, newName: String): Builder {
            require(srcCol !in keepMap) {
                "Column $srcCol already specified"
            }
            keepMap[srcCol] = ColumnId(newName, srcCol.clazz)
            return this
        }

        /**
         * The result should contain `srcCol` but in the output that data will have id `newId`.
         */
        fun <T : Any> keepAndRename(srcCol: ColumnId<T>, newId: ColumnId<out T>): Builder {
            require(srcCol !in keepMap) {
                "Column $srcCol already specified"
            }
            keepMap[srcCol] = newId
            return this
        }

        fun build(): ColumnSubsetTransform {
            return ColumnSubsetTransform(this)
        }
    }
}
