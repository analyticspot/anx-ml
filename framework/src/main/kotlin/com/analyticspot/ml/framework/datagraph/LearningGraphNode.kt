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
import com.analyticspot.ml.framework.datatransform.LearningTransform
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 * A [GraphNode] which takes a single input [DataSet] and applies a [LearningTransform] to it.
 */
class LearningGraphNode(builder: Builder) : HasTransformGraphNode<LearningTransform>(builder) {
    override val transform: LearningTransform = builder.transform

    companion object {
        fun build(id: Int, init: Builder.() -> Unit): LearningGraphNode {
            return with(Builder(id)) {
                init()
                build()
            }
        }
    }

    override fun getExecutionManager(parent: GraphExecution, execType: ExecutionType): NodeExecutionManager =
            ExecutionManager(this, execType, parent)

    class Builder(id: Int) : GraphNode.Builder(id) {
        lateinit var transform: LearningTransform

        fun build(): LearningGraphNode = LearningGraphNode(this)
    }

    // The execution manager for this node. Since this expects only a single input it signals onReadyToRun as soon as
    // onDataAvailable is called.
    private class ExecutionManager(
            override val graphNode: LearningGraphNode,
            private val execType: ExecutionType,
            parent: GraphExecution) : SingleInputExecutionManager(parent) {

        override fun doRun(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
            if (execType == ExecutionType.TRANSFORM) {
                return graphNode.transform.transform(dataSet, exec)
            } else {
                check(execType == ExecutionType.TRAIN_TRANSFORM)
                return graphNode.transform.trainTransform(dataSet, exec)
            }
        }
    }
}
