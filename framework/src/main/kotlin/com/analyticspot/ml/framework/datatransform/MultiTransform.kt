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

import com.analyticspot.ml.framework.dataset.DataSet
import java.util.concurrent.CompletableFuture

/**
 * A [MultiTransform] is like a [SingleDataTransform] but it takes more than one [DataSet] as input. It is recommended
 * to minimize use of [MultiTransform] as they are more complicated to write and much more complex to correctly
 * deserialize (due to issues with mapping [ValueId] to [ValueToken]. For the most part users should be able to write
 * regular [DataTransform] nodes and then use [MergeTransform] when inputs from multiple [DataSet]s are required.
 */
interface MultiTransform : DataTransform {
    companion object {
        /**
         * Used with `@JacksonInject` to indicate where you want the list of source
         * [DataSet] to be injected if using [StandardJsonFormat].
         */
        const val JSON_SOURCE_INJECTION_ID = "MultiTransformSources"
    }
    /**
     * Like [SingleDataTransform.transform] but takes a list of [DataSet] as input.
     */
    fun transform(dataSets: List<DataSet>): CompletableFuture<DataSet>
}
