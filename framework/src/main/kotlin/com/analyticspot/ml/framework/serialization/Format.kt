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

package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datagraph.GraphNode
import com.analyticspot.ml.framework.datatransform.DataTransform
import java.io.InputStream
import java.io.OutputStream

/**
 * A [Format] defines how a [DataTransform] is serialized and deserialized. It also defined [FormatMetaData] which is
 * used to inform the [Format] about the specifics of the [DataTransform] to be deserialized. The [FormatMetaData] is
 * specific to the [Format] and is serialized in the `graph.json` file in the zip. The data for an individual
 * [DataTransform] is in its own file in the zip. Thus, for example, [deserialize] is called with an `InputStream`
 * containing the file for the [DataTransform] and the [MetaData] that corresponds to the data found therein.
 */
interface Format<MetaDataT : FormatMetaData> : TransformFactory<MetaDataT> {
    val metaDataClass: Class<MetaDataT>
    /**
     * Returns a JSON-serializable object that is the metadata for the format.
     */
    fun getMetaData(transform: DataTransform) : MetaDataT

    /**
     * Saves the transform the provided `OutputStream`.
     */
    fun serialize(transform: DataTransform, output: OutputStream)

    override fun deserialize(metaData: MetaDataT, sources: List<GraphNode>, input: InputStream): DataTransform
}
