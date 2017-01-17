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

package com.analyticspot.ml.framework.description

import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonProperty

/**
 * Describes the outputs produced by an execution of the node. This allows each [GraphNode] to tell subscribing nodes
 * what columns are available.
 *
 * @param columns These are the columns **that can be known before training**. Some transforms can not know how many
 *     columns they will produce until they have been trained (e.g. a bag-of-words transform can't know how many words
 *     are in the vocabulary until it's been trained).
 * @param columnGroups These allow transforms which can't know how many columns they will produce to provide a way to
 *     reference the full set that will be produced. For example, a bag-of-words transform can't know the vocabulary
 *     size until it's trained, but it knows it will produce one column for each unique word in the input and can thus
 *     provide a [ColumnIdGroup] to allow access to all the generated columns.
 */
data class TransformDescription @JsonCreator constructor(@JsonProperty("columns") val columns: List<ColumnId<*>>,
        @JsonProperty("columnGroups") val columnGroups: List<ColumnIdGroup<*>> = listOf()) {
}
