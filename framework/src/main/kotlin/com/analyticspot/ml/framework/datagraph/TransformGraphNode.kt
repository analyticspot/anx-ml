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
import com.analyticspot.ml.framework.datatransform.SingleDataTransform
import org.slf4j.LoggerFactory
import java.util.concurrent.ExecutorService

/**
 * A [GraphNode] that takes a single input, runs it through a [DataTransform] and produces a single output.
 */
internal open class TransformGraphNode protected constructor(builder: Builder)
    : HasTransformGraphNode<SingleDataTransform>(builder) {
    override val transform: SingleDataTransform = builder.transform ?:
            throw IllegalArgumentException("Transform can not be null")

    companion object {
        private val log = LoggerFactory.getLogger(Companion::class.java)

        /**
         * Construct a [TransformGraphNode] by using the Kotlin builder pattern.
         */
        fun build(id: Int, init: Builder.() -> Unit): TransformGraphNode {
            return with(Builder(id)) {
                init()
                build()
            }
        }
    }

    override fun getExecutionManager(parent: GraphExecutionProtocol, execType: ExecutionType): NodeExecutionManager =
            ExecutionManager(this, parent)

    open class Builder(id: Int) : GraphNode.Builder(id) {
        var transform: SingleDataTransform? = null
            set(value) {
                field = value ?: throw IllegalArgumentException("Transform can not be null")
            }

        fun build(): TransformGraphNode = TransformGraphNode(this)
    }

    // The execution manager for this node. Since this expects only a single input it signals onReadyToRun as soon as
    // onDataAvailable is called.
    private class ExecutionManager(override val graphNode: TransformGraphNode, parent: GraphExecutionProtocol)
        : SingleInputExecutionManager(parent) {

        override fun doRun(dataSet: DataSet, exec: ExecutorService) = graphNode.transform.transform(dataSet, exec)
    }
}

