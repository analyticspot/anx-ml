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
import com.analyticspot.ml.framework.datatransform.SupervisedLearningTransform
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService
import java.util.concurrent.atomic.AtomicInteger

/**
 * A [GraphNode] that holds a [SupervisedLearningTransform].
 */
class SupervisedLearningGraphNode(builder: Builder) : HasTransformGraphNode<SupervisedLearningTransform>(builder) {
    override val transform: SupervisedLearningTransform = builder.transform

    // True if the target data comes from the same data set as the main data set (that is this is true if the
    // first and second parameters to trainTransform are the same). False otherwise.
    private val targetDataIsMainData: Boolean
        get() {
            check(sources.size == 1)
            check(trainOnlySources.size == 1 || trainOnlySources.size == 0)
            if (trainOnlySources.size == 0) {
                return true
            } else {
                check(trainOnlySources[0] != sources[0])
                return false
            }
        }

    companion object {
        // The value that will be used for subId in the subscription to the "main" data set.
        const val MAIN_DS_ID = 0
        // The value that will be used for subId in the subscription to the "target" data set.
        const val TARGET_DS_ID = 1

        fun build(id: Int, init: Builder.() -> Unit): SupervisedLearningGraphNode {
            return with(Builder(id)) {
                init()
                build()
            }
        }
    }

    override fun getExecutionManager(parent: GraphExecutionProtocol, execType: ExecutionType): NodeExecutionManager {
        return when (execType) {
            ExecutionType.TRANSFORM -> TransformExecutionManager(this, parent)
            ExecutionType.TRAIN_TRANSFORM -> {
                if (targetDataIsMainData) {
                    return SameDataTrainingExecutionManager(this, parent)
                } else {
                    return DifferentDataTrainingExecutionManager(this, parent)
                }
            }
        }
    }

    class Builder(id: Int) : GraphNode.Builder(id) {
        lateinit var transform: SupervisedLearningTransform

        fun build(): SupervisedLearningGraphNode = SupervisedLearningGraphNode(this)
    }

    // An execution manager for the training phase when the training and target data are different (so that we need to
    // receive two calls to onDataAvailable before we can run).
    private class DifferentDataTrainingExecutionManager(override val graphNode: SupervisedLearningGraphNode,
            private val parent: GraphExecutionProtocol) : NodeExecutionManager {
        private val numReceived = AtomicInteger(0)
        @Volatile
        private var mainData: DataSet? = null
        @Volatile
        private var targetData: DataSet? = null

        override fun onDataAvailable(subId: Int, data: DataSet) {
            if (subId == MAIN_DS_ID) {
                check(mainData == null)
                mainData = data
            } else {
                check(subId == TARGET_DS_ID)
                check(targetData == null)
                targetData = data
            }
            if (numReceived.incrementAndGet() >= 2) {
                assert(numReceived.get() == 2)
                assert(mainData != null)
                assert(targetData != null)
                parent.onReadyToRun(this)
            }
        }

        override fun run(exec: ExecutorService): CompletableFuture<DataSet> {
            assert(mainData != null)
            assert(targetData != null)
            return graphNode.transform.trainTransform(mainData!!, targetData!!, exec)
                    .whenComplete { dataSet, throwable ->
                        // Free the data sets so they can be GC'd
                    mainData = null
                    targetData = null
                }
        }
    }

    // An execution manager for the training phase when the training and target data the same. For example, the source
    // data set might contain the data and the target in which case there's only 1 data set needed before we can run.
    private class SameDataTrainingExecutionManager(override val graphNode: SupervisedLearningGraphNode,
            private val parent: GraphExecutionProtocol) : NodeExecutionManager {
        private val numReceived = AtomicInteger(0)
        @Volatile
        private var data: DataSet? = null

        override fun onDataAvailable(subId: Int, data: DataSet) {
            check(subId == MAIN_DS_ID)
            check(this.data == null)
            this.data = data
            parent.onReadyToRun(this)
        }

        override fun run(exec: ExecutorService): CompletableFuture<DataSet> {
            assert(data != null)
            return graphNode.transform.trainTransform(data!!, data!!, exec).whenComplete { dataSet, throwable ->
                // Free the data sets so they can be GC'd
                data = null
            }
        }
    }

    // An execution manager for performing a transform. This is just a regular single input data set.
    private class TransformExecutionManager(override val graphNode: SupervisedLearningGraphNode,
            parent: GraphExecutionProtocol) : SingleInputExecutionManager(parent) {
        override fun doRun(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
            return graphNode.transform.transform(dataSet, exec)
        }

    }
}
