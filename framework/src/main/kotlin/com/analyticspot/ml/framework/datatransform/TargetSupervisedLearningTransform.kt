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

package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.Column
import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import com.fasterxml.jackson.annotation.JsonIgnore
import java.util.concurrent.CompletableFuture

/**
 * As supervised learning algorithms that work with a single target value are common this is a convenience class that
 * extracts the target for the user. Subclasses then implmement the [trainTransform] overload that takes a `TargetT`
 * rather than the one that takes an entire [DataSet].
 *
 * @param targetColumn the token used to extract the target value from the data. Can be null as it will not be available
 *      when deserializing a trained transform.
 * @param <TargetT> the type of the target.
 */
abstract class TargetSupervisedLearningTransform<TargetT : Any>(@JsonIgnore val targetColumn: ColumnId<TargetT>?)
    : SupervisedLearningTransform {
    override final fun trainTransform(dataSet: DataSet, trainDs: DataSet): CompletableFuture<DataSet> {
        // targetColumn must be non-null for training but could be null for transform.
        return trainTransform(dataSet, trainDs.column(targetColumn!!))
    }

    abstract fun trainTransform(dataSet: DataSet, target: Column<TargetT?>): CompletableFuture<DataSet>

}
