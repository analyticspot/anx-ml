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
 * A [Format] defines how a [DataTransform] is serialized and deserialized.
 *
 * Each format's [getMetaData] method generally returns a different subclass of [FormatMetaData]. This sublcass contains
 * format-specific information about how the class is to be deserialized. However, this class is not generic as that
 * makes some kinds of formats impossible (e.g. because the [DelegatingFormat] can't know the type of the metadata it
 * will return statically).
 */
interface Format : TransformFactory {
    /**
     * Returns a JSON-serializable object that is the metadata for the format.
     */
    fun getMetaData(transform: DataTransform, serDeser: GraphSerDeser): FormatMetaData

    /**
     * Saves the transform the provided `OutputStream`. The `serDeser` argument will hold a reference to the
     * [GraphSerDeser] instance currently serializing the graph. This allows you to delegate serialization back to the
     * [GraphSerDeser].
     */
    fun serialize(transform: DataTransform, serDeser: GraphSerDeser, output: OutputStream)

    /**
     * Deserializes the transform from the provided `InputStream`. The `serDeser` argument will hold a reference to the
     * [GraphSerDeser] instance currently deserializing the graph. This allows you to delegate deserialization back to
     * the [GraphSerDeser].
     */
    override fun deserialize(metaData: FormatMetaData, sources: List<GraphNode>,
            serDeser: GraphSerDeser, input: InputStream): DataTransform
}
