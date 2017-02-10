package com.analyticspot.ml.framework.dataset

import com.analyticspot.ml.framework.feature.FeatureId

/**
 * A [DataSet] in which all [ColumnId]s are of type [FeatureId]. These data sets are what is generally passed to
 * learning algorithms.
 */
class FeatureDataSet private constructor(idAndColumns: Array<IdAndColumn<*, FeatureId<*>>>)
    : DataSetBase<FeatureId<*>>(idAndColumns) {
    override val columnIds: Array<FeatureId<*>> = idAndColumns.map { it.id }.toTypedArray()

    companion object {
        fun builder(): Builder = Builder()

        fun build(init: Builder.() -> Unit): FeatureDataSet {
            return with(Builder()) {
                init()
                build()
            }
        }

        /**
         * Creates a [DataSet] with a single column.
         */
        fun <T : Any> create(colId: FeatureId<T>, column: Column<T?>): FeatureDataSet {
            return build {
                addColumn(colId, column)
            }
        }

        /**
         * Creates a [DataSet] with a single column.
         */
        fun <T : Any> create(colId: FeatureId<T>, column: List<T?>): FeatureDataSet {
            return create(colId, ListColumn(column))
        }
    }

    class Builder {
        private val columns = mutableListOf<DataSetBase.IdAndColumn<*, FeatureId<*>>>()

        fun <ColT : Any> addColumn(id: FeatureId<ColT>, col: Column<ColT?>): Builder {
            columns += IdAndColumn(id, col)
            return this
        }

        fun <ColT : Any> addColumn(id: FeatureId<ColT>, col: List<ColT?>): Builder {
            return addColumn(id, ListColumn(col))
        }

        /**
         * Adds all the columns in `dataSet` to the new []DataSet].
         */
        fun addAll(dataSet: FeatureDataSet): Builder {
            columns += dataSet.columns.zip(dataSet.columnIds).map { IdAndColumn(it.second, it.first) }
            return this
        }

        fun build(): FeatureDataSet {
            columns.sort()
            return FeatureDataSet(columns.toTypedArray())
        }
    }
}
