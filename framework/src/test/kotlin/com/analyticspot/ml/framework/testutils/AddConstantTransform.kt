package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.datatransform.SingleItemDataTransform
import com.analyticspot.ml.framework.description.TransformDescription
import com.fasterxml.jackson.annotation.JacksonInject
import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonProperty

/**
 * Adds a constant to all integer values.
 *
 * @param toAdd the amount to add to each input.
 * @param srcDesc the [TransformDescription] for the source [GraphNode].
 */
class AddConstantTransform(val toAdd: Int, srcDesc: TransformDescription)
    : SingleItemDataTransform<Int, Int>(srcDesc, Int::class, Int::class) {

    @JsonCreator
    constructor(@JsonProperty("toAdd") toAdd: Int, @JacksonInject source: GraphNode)
            : this(toAdd, source.transformDescription)


    override fun transformItem(input: Int): Int {
        return toAdd + input
    }
}
