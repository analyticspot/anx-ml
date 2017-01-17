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

package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datagraph.GraphNode
import com.analyticspot.ml.framework.datatransform.DataTransform
import java.io.InputStream

/**
 * A function that can produce a [DataTransform] from serialized data. Registered with [GraphSerDeser] to do injection.
 */
interface TransformFactory<MetaDataT : FormatMetaData> {
    /**
     * Reads the data from `input` and deserializes it into a [DataTransform]. The `sources` indicate where the data
     * consumed by the [DataTransform] is produced and can be used to convert [ValueId] data into [ValueToken]
     * instances. The `metaData` describes what is found in the `input` and may help with deserialization.
     */
    fun deserialize(metaData: MetaDataT, sources: List<GraphNode>, input: InputStream): DataTransform
}
