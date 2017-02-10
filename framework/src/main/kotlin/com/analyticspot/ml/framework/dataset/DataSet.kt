package com.analyticspot.ml.framework.dataset

import com.analyticspot.ml.framework.description.ColumnId
// Lint disable as this is used but there's a ktlint bug.
import com.analyticspot.ml.utils.isAssignableFrom // ktlint-disable no-unused-imports


/**
 * A [DataSetBase] for any data type. If all the columns are a subclass of [FeatureId] use [FeatureDataSet] instead.
 */
class DataSet private constructor(idAndColumns: Array<IdAndColumn<*, ColumnId<*>>>)
    : DataSetBase<ColumnId<*>>(idAndColumns) {
    override val columnIds: Array<ColumnId<*>> = idAndColumns.map { it.id }.toTypedArray()

    companion object {
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

    class Builder {
        private val columns = mutableListOf<DataSetBase.IdAndColumn<*, ColumnId<*>>>()

        fun <ColT : Any> addColumn(id: ColumnId<ColT>, col: Column<ColT?>): Builder {
            columns += IdAndColumn(id, col)
            return this
        }

        fun <ColT : Any> addColumn(id: ColumnId<ColT>, col: List<ColT?>): Builder {
            return addColumn(id, ListColumn(col))
        }

        /**
         * Adds all the columns in `dataSet` to the new []DataSet].
         */
        fun <DsT : DataSetBase<*>> addAll(dataSet: DsT): Builder {
            columns += dataSet.columns.zip(dataSet.columnIds).map { IdAndColumn(it.second, it.first) }
            return this
        }

        fun build(): DataSet {
            columns.sort()
            return DataSet(columns.toTypedArray())
        }
    }
}
