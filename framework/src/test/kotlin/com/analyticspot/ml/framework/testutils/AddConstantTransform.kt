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

import com.analyticspot.ml.framework.datatransform.SingleItemDataTransform
import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonProperty

/**
 * Adds a constant to all integer values.
 *
 * @param toAdd the amount to add to each input.
 * @param srcDesc the [TransformDescription] for the source [GraphNode].
 */
class AddConstantTransform @JsonCreator constructor(@JsonProperty("toAdd") val toAdd: Int)
    : SingleItemDataTransform<Int, Int>(Int::class, Int::class) {

    override fun transformItem(input: Int): Int {
        return toAdd + input
    }
}
