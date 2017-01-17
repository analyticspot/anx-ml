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

import com.analyticspot.ml.framework.dataset.Column
import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.description.ColumnIdGroup
import com.analyticspot.ml.framework.description.TransformDescription
// Lint disable as this is used but there's a ktlint bug.
import com.analyticspot.ml.utils.isAssignableFrom // ktlint-disable no-unused-imports
import org.slf4j.LoggerFactory
import java.util.concurrent.CompletableFuture
import kotlin.reflect.KClass

/**
 * An abstract base class for [DataTransform] instances that can work on a single item at a time. Subclasses indicate
 * the data type they can work with. The base class will then operate on all columns such that the data in that column
 * is compatible with (as per `Class.isAssignableFrom`) the requested data type. For each such column [transformItem]
 * will be called.
 *
 * This will generate a [DataSet] with one column for each column that was in the original data set. If the column is
 * compatible with the requested data type the column will be transformed and the new column will have the same name but
 * the data type will be `OutputT`.
 *
 * @param <InputT> The type of the data the transform can process
 * @param <OutputT> The type of the data the transform produces
 *
 * @param srcTransDescription the [TransformDescription] for the source data.
 * @param inType the type of the input data this transform can handle.
 * @param outType the type of the output this will produce.
 */
abstract class SingleItemDataTransform<InputT : Any, OutputT : Any>(
        val srcTransDescription: TransformDescription,
        private val inType: KClass<InputT>,
        private val outType: KClass<OutputT>) : SingleDataTransform {

    companion object {
        private val log = LoggerFactory.getLogger(SingleItemDataTransform::class.java)
    }

    override final val description: TransformDescription by lazy {
        val newCols = srcTransDescription.columns.map {
            if (inType isAssignableFrom it.clazz) {
                ColumnId(it.name, outType)
            } else {
                it
            }
        }
        val newColGroups = srcTransDescription.columnGroups.map {
            if (inType isAssignableFrom it.clazz) {
                ColumnIdGroup(it.prefix, outType)
            } else {
                it
            }
        }

        TransformDescription(newCols, newColGroups)
    }

    final override fun transform(dataSet: DataSet): CompletableFuture<DataSet> {
        val resultBuilder = DataSet.builder()
        for (colId in dataSet.columnIds) {
            if (inType isAssignableFrom colId.clazz) {
                @Suppress("UNCHECKED_CAST")
                val col: Column<InputT> = dataSet.column(colId) as Column<InputT>
                log.debug("Column {} has type {} which is compatible with {} so will transforn it.",
                        colId.name, colId.clazz, inType)
                val newCol = col.mapToColumn { this.transformItem(it) }
                resultBuilder.addColumn(ColumnId(colId.name, outType), newCol)
            } else {
                log.debug("Column {} has type {} which is not compatible with {}. It will not be transformed",
                        colId.name, colId.clazz, inType)
                // Note that we **know** that the types of colId and dataSet.column match so the addColumn call is safe
                // but the compiler isn't smart enough...
                @Suppress("UNCHECKED_CAST")
                resultBuilder.addColumn(colId as ColumnId<Any>, dataSet.column(colId))
            }
        }
        return CompletableFuture.completedFuture(resultBuilder.build())
    }

    abstract fun transformItem(input: InputT): OutputT
}
