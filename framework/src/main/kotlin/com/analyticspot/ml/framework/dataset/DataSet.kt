package com.analyticspot.ml.framework.dataset

import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.utils.isAssignableFrom
import org.slf4j.LoggerFactory

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
    // Note: We want a very compact and iteratble format for the columns and their ids with fast random access and the
    // ability to do binary search by ColumnId. The built-in binarySearch methods don't allow you to compare two
    // different types so we can't have an array of IdAndColumn and search by ColumnId. So we break it apart into
    // the column ids and the columns. We can then binary search the ids to get the index into the columns. In the
    // future it might be worth combining these again and writing our own binary search.
    val columnIds: Array<ColumnId<*>> = idAndColumns.map { it.id }.toTypedArray()

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
        assert(isLegal(idAndColumns)) {
            throw IllegalArgumentException("Columns are not legal")
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
        fun <T : Any> create(colId: ColumnId<T>, column: Column<T?>): DataSet {
            return build {
                addColumn(colId, column)
            }
        }

        /**
         * Creates a [DataSet] with a single column.
         */
        fun <T : Any> create(colId: ColumnId<T>, column: List<T?>): DataSet {
            return create(colId, ListColumn(column))
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
    }

    // Column names must be unique, be in sorted order, and all columns must have the same length. This checks to make
    // sure that's the case. Note that this is somewhat expensive so this should probably always be wrapped in an
    // `assert` call.
    private fun isLegal(cols: Array<IdAndColumn<*>>): Boolean {
        if (cols.size == 0) {
            return true
        }

        var lastCol = cols[0]
        val colSize = lastCol.column.size
        for (idx in 1 until cols.size) {
            val nextCol = cols[idx]
            if (lastCol > nextCol) {
                log.error("Columns are not in sorted order.")
                return false
            }
            if (lastCol.id.name == nextCol.id.name) {
                log.error("Duplicate columns with name {}", lastCol.id.name)
                return false
            }
            if (nextCol.column.size != colSize) {
                log.error("Not all columns are the same length")
                return false
            }
            lastCol = nextCol
        }
        return true
    }

    fun <T : Any> value(columnId: ColumnId<T>, rowIdx: Int): T? {
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

    private data class IdAndColumn<out ColumnT : Any>(val id: ColumnId<out ColumnT>, val column: Column<ColumnT?>)
        : Comparable<IdAndColumn<*>> {
        override fun compareTo(other: IdAndColumn<*>): Int {
            return id.name.compareTo(other.id.name)
        }
    }

    class Builder {
        private val columns = mutableListOf<IdAndColumn<*>>()

        fun <ColT : Any> addColumn(id: ColumnId<ColT>, col: Column<ColT?>): Builder {
            columns += IdAndColumn(id, col)
            return this
        }

        /**
         * Adds all the columns in `dataSet` to the new []DataSet].
         */
        fun addAll(dataSet: DataSet): Builder {
            columns += dataSet.columns.zip(dataSet.columnIds).map { IdAndColumn(it.second, it.first) }
            return this
        }

        fun build(): DataSet {
            columns.sort()
            return DataSet(columns.toTypedArray())
        }
    }
}

