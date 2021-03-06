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
import com.analyticspot.ml.framework.description.ColumnIdGroup
import com.analyticspot.ml.framework.metadata.ColumnMetaData
import com.analyticspot.ml.framework.serialization.JsonMapper
// Lint disable as this is used but there's a ktlint bug.
import com.analyticspot.ml.utils.isAssignableFrom // ktlint-disable no-unused-imports
import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonIgnore
import com.fasterxml.jackson.annotation.JsonProperty
import org.slf4j.LoggerFactory
import java.io.File
import java.io.InputStream
import java.io.OutputStream
import java.util.Collections
import java.util.Random

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
class DataSet private constructor(idAndColumns: Array<IdAndColumn<*>>) {
    /**
     * All of the columns ids in this [DataSet].
     */
    // Note: We want a very compact and iterable format for the columns and their ids with fast random access and the
    // ability to do binary search by ColumnId. The built-in binarySearch methods don't allow you to compare two
    // different types so we can't have an array of IdAndColumn and search by ColumnId. So we break it apart into
    // the column ids and the columns. We can then binary search the ids to get the index into the columns. In the
    // future it might be worth combining these again and writing our own binary search.
    val columnIds: Array<ColumnId<*>> = idAndColumns.map { it.id }.toTypedArray()

    /**
     * A dataset may, optionally, contain metadata about one or more columns in the data set. This is a map from the
     * name of the column (that is [ColumnId.name]) to the metadata about the column. Metadata is typical things like
     * categorical features in which case it indicates the range of possible values for the feature.
     */
    val metaData: Map<String, ColumnMetaData> = idAndColumns
            .filter { it.metaData != null }
            .associate { it.id.name to it.metaData!! }

    /**
     * All of the columns in this [DataSet]
     */
    val columns: Array<Column<*>> = idAndColumns.map { it.column }.toTypedArray()

    @get:JsonIgnore
    val numRows: Int
        get() {
            if (columns.size == 0) {
                return 0
            } else {
                return columns[0].size
            }
        }

    @get:JsonIgnore
    val numColumns: Int
        get() = columnIds.size

    init {
        val colProblems = columnErrors(idAndColumns)
        if (colProblems != null) {
            throw IllegalArgumentException(colProblems)
        }
    }

    companion object {
        private val log = LoggerFactory.getLogger(DataSet::class.java)

        fun builder(): Builder = Builder()

        fun build(init: Builder.() -> Unit): DataSet {
            return with(Builder()) {
                init()
                build()
            }
        }

        /**
         * Creates a [DataSet] with a single column.
         */
        fun <T : Any> create(colId: ColumnId<T>, column: Column<T?>, metaData: ColumnMetaData? = null): DataSet {
            return build {
                addColumn(colId, column, metaData)
            }
        }

        /**
         * Creates a [DataSet] with a single column.
         */
        fun <T : Any> create(colId: ColumnId<T>, column: List<T?>, metaData: ColumnMetaData? = null): DataSet {
            return create(colId, ListColumn(column), metaData)
        }

        /**
         * Given a "matrix" of values such that `data[0]` is the first row of data, `data[1]` is the second row, etc.
         * this builds a [DataSet]. This also checks that everything is "legal" - all rows have the same number of
         * columns, the data types in each column match the data types specified in `colIds`, etc.
         *
         * Note that this is fairly inefficient as it is dynamically creating lists for the columns and they have
         * unknown size. Also, the type checking takes time. This is a convenience method for small data sets, manual
         * testing, etc. but shouldn't be used for loading very large sets of data.
         */
        fun fromMatrix(colIds: List<ColumnId<out Any>>, data: List<List<Any?>>): DataSet {
            val colLists = Array<MutableList<Any?>>(colIds.size) {
                mutableListOf()
            }

            data.forEachIndexed { rowNum, row ->
                require(row.size == colIds.size) {
                    "Row $rowNum contains ${row.size} columns but ${colIds.size} column Ids were provided"
                }
                row.forEachIndexed { colNum, value ->
                    if (value == null || colIds[colNum].clazz isAssignableFrom value.javaClass) {
                        colLists[colNum].add(value)
                    } else {
                        throw IllegalArgumentException("Row $rowNum, column $colNum is of type ${value.javaClass} " +
                                "which is not compatible with ${colIds[colNum].clazz}")
                    }
                }
            }

            return build {
                colIds.zip(colLists).forEach {
                    @Suppress("UNCHECKED_CAST")
                    addColumn(it.first as ColumnId<Any>, ListColumn(it.second))
                }
            }
        }

        /**
         * Constructs a [DataSet] from data previously saved via [saveToString]. Note that there is minimal type
         * checking or other validation here: this should only be used when you trust that the serialized data is
         * valid.
         */
        @JvmStatic
        fun fromSaved(data: String): DataSet {
            return JsonMapper.mapper.readValue(data, DataSet::class.java)
        }

        /**
         * Like the other [fromSaved] overload but used an `InputStream`.
         */
        @JvmStatic
        fun fromSaved(input: InputStream): DataSet {
            return JsonMapper.mapper.readValue(input, DataSet::class.java)
        }

        // This is just for Jackson to use as a constructor function.
        @JsonCreator
        @JvmStatic
        private fun fromSerialized(
                @JsonProperty("columnIds") columnIds: List<ColumnId<*>>,
                @JsonProperty("metaData") metaData: Map<String, ColumnMetaData>,
                @JsonProperty("columns") columns: List<Column<*>>): DataSet {
            require(columnIds.size == columns.size)
            val idAndColumn = Array<IdAndColumn<*>>(columnIds.size) { idx ->
                IdAndColumn(columnIds[idx], columns[idx], metaData[columnIds[idx].name])
            }
            return DataSet(idAndColumn)
        }
    }

    // Column names must be unique, be in sorted order, and all columns must have the same length. This checks to make
    // sure that's the case. This returns a non-null string indicating what the issue is if the columns aren't legal.
    // Returns null if everything is legal.
    private fun columnErrors(cols: Array<IdAndColumn<*>>): String? {
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

    /**
     * Returns all [ColumnIds] instances for the given group.
     */
    fun <ColT : Any> colIdsInGroup(group: ColumnIdGroup<ColT>, clazz: Class<ColT>): List<ColumnId<ColT>> {
        val fullPrefix = group.prefix + ColumnId.GROUP_SEPARATOR
        // Create a fake ColumnId that will sort to the very beginning of the ColumnId array. We don't expect the actual
        // value to be found but `binarySearch` returns (-insertion point - 1) when the value is not found where
        // insertion point is defined as the index at which the element should be inserted, so that the list remains
        // sorted.
        val searchResult = columnIds.binarySearch(ColumnId(fullPrefix, Any::class))
        check(searchResult < 0) {
            "Did not expect to find a columnId whose name is the group prefix"
        }
        val firstMatchIdx = -1 * searchResult - 1
        val result = mutableListOf<ColumnId<ColT>>()
        for (colIdx in firstMatchIdx until columnIds.size) {
            val colId = columnIds[colIdx]
            if (colId.name.startsWith(fullPrefix)) {
                check(clazz.isAssignableFrom(colId.clazz))
                @Suppress("UNCHECKED_CAST")
                result.add(colId as ColumnId<ColT>)
            } else {
                break
            }
        }
        return result
    }

    /**
     * Returns a **new** [DataSet] that contains only the given columns. All metadata will be preserved.
     */
    fun columnSubset(vararg columns: ColumnId<*>): DataSet {
        val resultDs = DataSet.build {
            for (col in columns) {
                val meta = metaData[col.name]
                @Suppress("UNCHECKED_CAST")
                addColumn(col as ColumnId<Any>, column(col), meta)
            }
        }
        return resultDs
    }

    /**
     * Like the other [columnSubset] overload by takes a list of column ids.
     */
    fun columnSubset(columns: List<ColumnId<*>>): DataSet {
        return columnSubset(*columns.toTypedArray())
    }

    /**
     * Like the other [columnSubset] overload but takes just the column names instead of their full ids.
     */
    fun columnSubset(vararg columnNames: String): DataSet {
        val colIds = columnNames.map { columnIdWithNameUntyped(it) }
        return columnSubset(colIds)
    }

    /**
     * Returns a **new** [DataSet] that contains all the columns in this data set **except** those passed to this
     * function.
     */
    fun allColumnsExcept(columnNames: Set<String>): DataSet {
        return DataSet.build {
            columnIds.forEach { colId ->
                if (!columnNames.contains(colId.name)) {
                    @Suppress("UNCHECKED_CAST")
                    addColumn(colId as ColumnId<Any>, column(colId), metaData[colId.name])
                }
            }
        }
    }

    /**
     * Convenience overload version of [allColumnsExcept].
     */
    fun allColumnsExcept(vararg columnNames: String): DataSet {
        return allColumnsExcept(columnNames.toSet())
    }

    /**
     * Randomly divides the [DataSet] into 2 smaller [DataSet]s. The first contains a randomly selected set of `n` rows
     * and the 2nd contains all the rows that aren't in the first. Note that rows are sampled but **not** permuted.
     */
    fun randomSubsets(n: Int, rng: Random = Random()): Pair<DataSet, DataSet> {
        val indices = 0.until(numRows).toList()
        Collections.shuffle(indices, rng)
        val firstSubsetIndices = indices.subList(0, n).toSet()
        check(firstSubsetIndices.size == n)
        val secondSubsetIndices = indices.subList(n, indices.size).toSet()
        check(secondSubsetIndices.size == numRows - n)

        return Pair(rowsSubset(firstSubsetIndices), rowsSubset(secondSubsetIndices))
    }

    /**
     * The same as the other [randomSubsets] overload but you specify the **fraction** of rows you want in the first
     * [DataSet] rather than the number. The actual number of rows in the first set will be `floor(frac * numRows)`.
     */
    fun randomSubsets(frac: Float, rng: Random = Random()): Pair<DataSet, DataSet> {
        val n = (frac * numRows.toFloat()).toInt()
        return randomSubsets(n, rng)
    }

    /**
     * Returns a new [DataSet] that contains all the columns and metadata in this [DataSet] but only a subset of the
     * rows (the ones specified by `indices`).
     */
    fun rowsSubset(indices: Set<Int>): DataSet {
        return DataSet.build {
            columnIds.forEach {
                val md = metaData[it.name]
                val colData = column(it).filterIndexed { idx, item -> idx in indices }
                @Suppress("UNCHECKED_CAST")
                addColumn(it as ColumnId<Any>, colData, md)
            }
        }
    }

    /**
     * Returns a new [DataSet] that contains all the columns in this data set plus all the columns in `other`. For this
     * to work `other` must have the same number of rows as this and the column names must all be differentt. Note
     * that in Kotlin this will also be available via the `+` operator.
     */
    operator fun plus(other: DataSet): DataSet {
        require(other.numRows == numRows)
        return build {
            addAll(this@DataSet)
            addAll(other)
        }
    }

    /**
     * Returns a new [DataSet] that contains all the columns in this data set plus all the columns in `other` like
     * [plus]. However, unlike [plus] this will not throw an exception if `other` contains some columns that have the
     * same name as the columns here. Instead we will assume the columns are identical and just keep one of them. We
     * do some minimal checks to try and ensure the columns are identical but, in the interest of speed, we don't
     * do any deep checks (unless asserts are enabled).
     */
    fun combineAndSkipDuplicates(other: DataSet): DataSet {
        require(other.numRows == numRows)
        return build {
            columnIds.forEach {
                @Suppress("UNCHECKED_CAST")
                addColumn(it as ColumnId<Any>, column(it), metaData[it.name])
            }

            other.columnIds.forEach { colId ->
                if (columnIds.binarySearch(colId) < 0) {
                    @Suppress("UNCHECKED_CAST")
                    addColumn(colId as ColumnId<Any>, other.column(colId), other.metaData[colId.name])
                } else {
                    check(colId.clazz == columnIdWithNameUntyped(colId.name).clazz)
                    assert(other.column(colId).zip(column(colId)).all { it.first == it.second })
                    log.info("Skipping duplicate column {}", colId)
                }
            }
        }
    }

    /**
     * Convenience overload with reified type parameter.
     */
    inline fun <reified ColT : Any> colIdsInGroup(group: ColumnIdGroup<ColT>): List<ColumnId<ColT>> {
        return colIdsInGroup(group, ColT::class.java)
    }

    /**
     * Saves the [DataSet] as JSON to the given `output`. Unlike the [toDelimited] methods this also preserves the
     * column names, metadata, etc. You can use the corresponding [fromSaved] method to load a [DataSet] saved via this
     * method.
     */
    fun save(output: OutputStream) {
        val mapper = JsonMapper.mapper
        mapper.writeValue(output, this)
    }

    /**
     * Identical to [save] but instead of writing the JSON data to a file it returns it as a String.
     */
    fun saveToString(): String {
        return JsonMapper.mapper.writeValueAsString(this)
    }

    /**
     * Saves the [DataSet] to the passed `OutputStream` as CSV (or you can specify another delimiter via the `delimiter`
     * parameter). For this to work all column values must support `toString`. It is the responsibility of the caller to
     * close the `OutputStream`.
     *
     * @param output the `OutputStream` to which the data should be saved.
     * @param header if true a header row will be written before the data. The header will contain the [ColumnId.name]
     *     for each column. If false there will be no header row.
     * @param nullVal the value to write for missing values
     * @param delimiter the value to be written between the values. If it's a `,` you get a CSV file, if it's a `\t` you
     *     get a TSV file, etc.
     */
    fun toDelimited(output: OutputStream, header: Boolean = true, nullVal: String = "null", delimiter: String = ",") {
        val writer = output.writer()
        if (header) {
            writer.write(columnIds.map { it.name }.joinToString(delimiter))
            writer.write("\n")
        }

        for (row in 0 until numRows) {
            val strValues = columnIds.map { value(row, it)?.toString() ?: nullVal }
            writer.write(strValues.joinToString(delimiter))
            writer.write("\n")
        }
        writer.flush()
    }

    /**
     * The same as the other [toDelimited] overload but writes to a file `File` and closes the file when done.
     */
    fun toDelimited(output: File, header: Boolean = true, nullVal: String = "null", delimiter: String = ",") {
        val outStream = output.outputStream()
        toDelimited(outStream, header, nullVal, delimiter)
        outStream.close()
    }

    /**
     * The same as the other [toDelimited] overload but writes to a file at `filePath` and closes the file when done.
     */
    fun toDelimited(filePath: String, header: Boolean = true, nullVal: String = "null", delimiter: String = ",") {
        toDelimited(File(filePath), header, nullVal, delimiter)
    }

    /**
     * Returns the [ColumnId] with the given name. This method should be used sparingly because it is linear time.
     */
    inline fun <reified ColT : Any> columnIdWithName(name: String): ColumnId<ColT> {
        val colId = columnIdWithNameUntyped(name)
        require(colId.clazz isAssignableFrom ColT::class) {
            "columIdWithName was called specifying a type of ${ColT::class} but actual type was ${colId.clazz}"
        }
        @Suppress("UNCHECKED_CAST")
        return colId as ColumnId<ColT>
    }

    /**
     * Returns the columId with the given name. No type information about the column will be available.
     */
    fun columnIdWithNameUntyped(name: String): ColumnId<*> {
        val colId = columnIds.find { it.name == name }
        if (colId == null) {
            throw IllegalArgumentException("Column with name $name not found")
        } else {
            return colId
        }
    }

    private data class IdAndColumn<out ColumnT : Any>(
            val id: ColumnId<out ColumnT>,
            val column: Column<ColumnT?>,
            val metaData: ColumnMetaData? = null) : Comparable<IdAndColumn<*>> {
        override fun compareTo(other: IdAndColumn<*>): Int {
            return id.name.compareTo(other.id.name)
        }
    }

    class Builder {
        private val columns = mutableListOf<IdAndColumn<*>>()

        fun <ColT : Any> addColumn(
                id: ColumnId<ColT>,
                col: Column<ColT?>,
                metaData: ColumnMetaData? = null): Builder {
            columns += IdAndColumn(id, col, metaData)
            return this
        }

        fun <ColT : Any> addColumn(
                id: ColumnId<ColT>,
                col: List<ColT?>,
                metaData: ColumnMetaData? = null): Builder {
            return addColumn(id, ListColumn(col), metaData)
        }

        /**
         * Adds all the columns in `dataSet` to the new [DataSet]. If the columns contained metadata that will be
         * preserved.
         */
        fun addAll(dataSet: DataSet): Builder {
            dataSet.columnIds.forEach {
                val md = dataSet.metaData[it.name]
                columns += IdAndColumn(it, dataSet.column(it), md)
            }
            return this
        }

        fun build(): DataSet {
            columns.sort()
            return DataSet(columns.toTypedArray())
        }
    }
}

