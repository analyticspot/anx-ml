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

package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.dataset.ListColumn
import com.analyticspot.ml.framework.datatransform.LearningTransform
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.description.TransformDescription
import org.slf4j.LoggerFactory
import java.util.Collections
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 * Learns the minimum value for a single column and produces a [DataSet] with a single column, identified by `resultId`,
 * which contains that minimum value.
 */
class LearnMinTransform(private val srcColumn: ColumnId<Int>, val resultId: ColumnId<Int>) : LearningTransform {
    private var minValue: Int = Int.MAX_VALUE
    override val description: TransformDescription
        get() = TransformDescription(listOf(resultId))

    companion object {
        private val log = LoggerFactory.getLogger(LearnMinTransform::class.java)
    }

    override fun trainTransform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        log.debug("{} is training", this.javaClass)
        val dsMin = dataSet.column(srcColumn).map { it ?: Int.MAX_VALUE }.min()
        minValue = dsMin ?: minValue
        return transform(dataSet, exec)
    }

    override fun transform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        val col = ListColumn(Collections.nCopies(dataSet.numRows, minValue))
        val ds = DataSet.build {
            addColumn(resultId, col)
        }
        return CompletableFuture.completedFuture(ds)
    }
}
