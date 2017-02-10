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

package com.analyticspot.ml.framework.dataset

import com.analyticspot.ml.framework.description.ColumnId
// Lint disable as this is used but there's a ktlint bug.
import com.analyticspot.ml.utils.isAssignableFrom // ktlint-disable no-unused-imports

/**
 * This is what holds the data that is passed through the [DataGraph]. It is column-oriented so that creating a
 * [DataSet] that is a combination of columns from multiple [DataSet] instances or a new [DataSet] that is a subset of
 * the columns of an existing [DataSet] is very inexpensive.
 *
 * [DataSet] instances are immutable and safe to share between threads.
 *
 * @param <ColumnT> this is the **super type** of all columns in this [DataSet]. It is therefore not uncommon to have a
 * [DataSet] of type `Any` (in Kotlin) or `Object` (in Java). However, you can still have type-safe access to individual
 * columns and values because the [ColumnId] passed to methods like [value] can be any subclass of `ColumnT`.
 */
abstract class DataSetBase<ColIdT : ColumnId<*>> protected constructor(idAndColumns: Array<IdAndColumn<*, ColIdT>>) {
    /**
     * All of the columns ids in this [DataSet].
     */
    // Note: We want a very compact and iteratble format for the columns and their ids with fast random access and the
    // ability to do binary search by ColumnId. The built-in binarySearch methods don't allow you to compare two
    // different types so we can't have an array of IdAndColumn and search by ColumnId. So we break it apart into
    // the column ids and the columns. We can then binary search the ids to get the index into the columns. In the
    // future it might be worth combining these again and writing our own binary search.
    abstract val columnIds: Array<ColIdT>

    /**
     * All of the columns in this [DataSet]
     */
    val columns: Array<Column<*>> = idAndColumns.map { it.column }.toTypedArray()

    val numRows: Int
        get() {
            if (columns.size == 0) {
                return 0
            } else {
                return columns[0].size
            }
        }

    val numColumns: Int
        get() = columnIds.size

    init {
        val colProblems = columnErrors(idAndColumns)
        if (colProblems != null) {
            throw IllegalArgumentException(colProblems)
        }
    }

    // Column names must be unique, be in sorted order, and all columns must have the same length. This checks to make
    // sure that's the case. This returns a non-null string indicating what the issue is if the columns aren't legal.
    // Returns null if everything is legal.
    private fun columnErrors(cols: Array<IdAndColumn<*, ColIdT>>): String? {
        if (cols.size == 0) {
            return null
        }

        var lastCol = cols[0]
        val colSize = lastCol.column.size
        for (idx in 1 until cols.size) {
            val nextCol = cols[idx]
            if (lastCol > nextCol) {
                return "Columns are not in sorted order."
            }
            if (lastCol.id.name == nextCol.id.name) {
                return "Duplicate columns with name '${lastCol.id.name}'"
            }
            if (nextCol.column.size != colSize) {
                return "Not all columns are the same length"
            }
            lastCol = nextCol
        }
        return null
    }

    fun <T : Any> value(rowIdx: Int, columnId: ColumnId<T>): T? {
        val col = column(columnId)
        return col[rowIdx]
    }

    fun <T : Any> column(columnId: ColumnId<T>): Column<T?> {
        val colIdx = columnIds.binarySearch(columnId)
        check(colIdx >= 0) {
            "Column $columnId not found"
        }
        val theCol = columns[colIdx]
        val theColId = columnIds[colIdx]
        check(columnId.clazz isAssignableFrom theColId.clazz) {
            "Column ${columnId.name} is of type ${theColId.clazz} but passed ValueId has type ${columnId.clazz}"
        }
        @Suppress("UNCHECKED_CAST")
        return theCol as Column<T>
    }

    protected data class IdAndColumn<out ColumnT : Any, out ColIdT : ColumnId<out ColumnT>>(
            val id: ColIdT, val column: Column<ColumnT?>) : Comparable<IdAndColumn<*, *>> {
        override fun compareTo(other: IdAndColumn<*, *>): Int {
            return id.name.compareTo(other.id.name)
        }
    }
}

