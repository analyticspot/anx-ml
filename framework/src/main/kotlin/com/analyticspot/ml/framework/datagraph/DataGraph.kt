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
import com.analyticspot.ml.framework.datatransform.AllColumnsExceptTransform
import com.analyticspot.ml.framework.datatransform.ColumnSubsetTransform
import com.analyticspot.ml.framework.datatransform.LearningTransform
import com.analyticspot.ml.framework.datatransform.MergeTransform
import com.analyticspot.ml.framework.datatransform.MultiTransform
import com.analyticspot.ml.framework.datatransform.SingleDataTransform
import com.analyticspot.ml.framework.datatransform.SupervisedLearningTransform
import com.analyticspot.ml.framework.description.ColumnId
import org.slf4j.LoggerFactory
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 * A [DataGraph] is a directed acyclic graph of [GraphNode] objects. Typically each [GraphNode] represents a
 * [DataTransform]. Thus the graph edges are data sources and the nodes are transformations. The [source] [GraphNode]
 * is the root of the [DataGraph] and it where the original source data that is used for learning or prediciton comes
 * from. The [result] [GraphNode] is the final node; the value it produces is the final output of the entire graph. For
 * classification or regression problems the output of the [result] contain the prediction (and, perhaps, some
 * metadata).
 *
 * Note: Unit tests for this are in `GraphExecutionTest`.
 */
class DataGraph(builder: GraphBuilder) : LearningTransform {
    val source: SourceGraphNodeBase
    val result: GraphNode
    // An array of all the GraphNodes such that a node `x` can be found at `allNodes[x.id]`. Can be null because when
    // we deserialize a graph it won't contain nodes that were only required for training.
    internal val allNodes: Array<GraphNode?>

    init {
        source = builder.source
        result = builder.result

        val maxNodeId = builder.nodesById.keys.max() ?: throw IllegalStateException("Graph is empty.")
        allNodes = Array<GraphNode?>(maxNodeId + 1) {
            builder.nodesById[it]
        }
        if (builder.missingTrainNodes) {
            correctNonTrainGraph()
        } else {
            correctGraph()
        }
    }

    companion object {
        private val log = LoggerFactory.getLogger(Companion::class.java)
        /**
         * Kotlin-style builder for a [DataGraph].
         */
        inline fun build(init: GraphBuilder.() -> Unit): DataGraph {
            with(GraphBuilder()) {
                init()
                return build()
            }
        }

        @JvmStatic
        fun builder(): GraphBuilder = GraphBuilder()
    }

    // This is the method that walks back through the graph and updates sources and subscriptions so they are
    // correct. See the class level comments for details.
    private fun correctGraph() {
        // We go in reverse topological order here so that by the time we hit a node, X, we know that we've already
        // hit all the subscribers of X.
        for (node in sortWithTrainBackwards(this)) {
            if (node.subscribers.size == 0 && node.trainOnlySubscribers.size != 0) {
                // This node has subscribers but they are all train-only. Thus, all the sources of this node are really
                // train-only sources.
                node.trainOnlySources += node.sources
                node.sources = listOf()
            }

            for (subTo in node.sources) {
                subTo.source.subscribers += Subscription(node, subTo.subId)
            }

            for (subTo in node.trainOnlySources) {
                subTo.source.trainOnlySubscribers += Subscription(node, subTo.subId)
            }
        }
    }

    // Like correctGraph but when we're deserializing a graph so we know there's no train-only information.
    private fun correctNonTrainGraph() {
        for (node in sortBackwards(this)) {
            check(node.trainOnlySubscribers.size == 0)

            for (subTo in node.sources) {
                subTo.source.subscribers += Subscription(node, subTo.subId)
            }
        }
    }

    /**
     * Creates a [DataSet] compatible with the [source] defined for this graph from the passed values. This ensures that
     * the column types are compatible with the columns declared for [source] and that there are the right number of
     * columns. This assumes the source is not being used for training so it will fail if data is provided for
     * train-only columns.
     */
    fun createSource(vararg vals: Any?): DataSet {
        return ensureSourceGraphNodeAndRun {
            DataSet.fromMatrix(it.columnIds.minus(it.trainOnlyColumnIds), listOf(vals.asList()))
        }
    }

    /**
     * Like the other [createSource] overload but lets you pass an array of rows rather than just a single row.
     */
    fun createSource(data: List<List<Any?>>): DataSet {
        return ensureSourceGraphNodeAndRun {
            DataSet.fromMatrix(it.columnIds.minus(it.trainOnlyColumnIds), data)
        }
    }

    /**
     * Like the other [createSource] methods but lets you pass an array of arrays.
     */
    fun createSource(matrix: Array<Array<Any?>>): DataSet {
        return ensureSourceGraphNodeAndRun {
            val asLists = matrix.map { it.asList() }
            createSource(asLists)
        }
    }

    /**
     * Like [createSource] but includes train-only columns.
     */
    fun createTrainingSource(vararg vals: Any?): DataSet {
        return ensureSourceGraphNodeAndRun {
            DataSet.fromMatrix(it.columnIds, listOf(vals.asList()))
        }
    }

    /**
     * Like the other [createTrainingSource] methods but lets you pass an array of arrays.
     */
    fun createTrainingSource(matrix: Array<Array<Any?>>): DataSet {
        return ensureSourceGraphNodeAndRun {
            val asLists = matrix.map { it.asList() }
            createTrainingSource(asLists)
        }
    }

    // Makes sure the type of [source] is `SourceGraphNode`. If it's not, it throws. If it is, it runs the passed
    // function.
    private fun <R> ensureSourceGraphNodeAndRun(toRun: (SourceGraphNode) -> R): R {
        if (source is SourceGraphNode) {
            return toRun(source)
        } else {
            throw IllegalStateException("This only works if the graph's source is a SourceGraphNode. Source for " +
                    "this graph is ${source.javaClass}")
        }
    }

    /**
     * Like the other [createTrainingSource] overload but lets you pass an array of rows rather than just a single row.
     */
    fun createTrainingSource(data: List<List<Any?>>): DataSet {
        if (source is SourceGraphNode) {
            return DataSet.fromMatrix(source.columnIds, data)
        } else {
            throw IllegalStateException("You can't call the createSource method if the graph's source is a " +
                    "DataSetSourceGraphNode")
        }
    }

    /**
     * Run the data through the entire graph. The result type is a future of `DataSet` because the graph
     * might contain an asynchronous [DataTransform] or it might contain an [OnDemandValue].
     */
    override fun transform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        val graphExec = GraphExecution(this, ExecutionType.TRANSFORM, exec)
        return graphExec.execute(dataSet)
    }

    /**
     * Train the graph on the given inputs and return the result of transforming the source data set via the trained
     * transformers.
     */
    override fun trainTransform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        val graphExec = GraphExecution(this, ExecutionType.TRAIN_TRANSFORM, exec)
        return graphExec.execute(dataSet)
    }

    /**
     * Builds a [DataGraph]. Note that during the building process subscription information is not set. Similarly,
     * [GraphNode.sources] and [GraphNode.trainOnlySources] may be incorrect. This is because if we add a node A we
     * can't know if the subscribers to A will be "regular" or "train-only". Thus, in the [build] method we walk
     * backward through the graph and update the train-only sources/subscribers information so that it is correct.
     * This is important as we only serialize the "regular" (non-train-only) parts of the graph. That way unnecessary
     * parts of the graph don't execute and we don't require the user to provide train-only data for the source in
     * order to call [DataGraph.transform].
     */
    class GraphBuilder {
        internal lateinit var source: SourceGraphNodeBase

        lateinit var result: GraphNode

        // When we're deserializing a graph the train-only nodes will be missing and this will be true
        internal var missingTrainNodes: Boolean = false

        // An array of GraphNode such that nodesById[idx] returns the GraphNode whose id is idx.
        internal val nodesById: MutableMap<Int, GraphNode> = mutableMapOf()

        internal var nextId = 0

        fun setSource(init: SourceGraphNode.Builder.() -> Unit): SourceGraphNode {
            val sourceNode = SourceGraphNode.build(nextId++, init)
            return setSource(sourceNode)
        }

        /**
         * Specify the format of the node that is the source for this graph using a Java-style builder.
         */
        fun source(): SourceBuilder = SourceBuilder(nextId++)

        internal fun setSource(node: SourceGraphNode): SourceGraphNode {
            source = node
            check(!nodesById.containsKey(node.id))
            nodesById[node.id] = node
            return node
        }

        /**
         * Specify that the source for this graph is a [DataSetSourceGraphNode].
         */
        fun dataSetSource(): DataSetSourceGraphNode {
            return setDataSetSource(DataSetSourceGraphNode(nextId++))
        }

        internal fun setDataSetSource(node: DataSetSourceGraphNode): DataSetSourceGraphNode {
            source = node
            check(!nodesById.containsKey(source.id))
            nodesById[source.id] = source
            return node
        }

        fun addTransform(src: GraphNode, transform: SingleDataTransform): GraphNode {
            check(!(transform is SupervisedLearningTransform)) {
                "This is the wrong addTransform overload to call for a supervised learning algorithm."
            }
            return addTransform(src, transform, nextId++)
        }

        internal fun addTransform(src: GraphNode, transform: SingleDataTransform, nodeId: Int): GraphNode {
            log.debug("Adding an untrained transform to the graph.")
            val node = TransformGraphNode.build(nodeId) {
                this.transform = transform
                sources += SubscribedTo(src, 0)
            }
            addNodeToGraph(node)
            return node
        }

        fun addTransform(src: GraphNode, transform: LearningTransform): GraphNode {
            check(!(transform is SupervisedLearningTransform)) {
                "This is the wrong addTransform overload to call for a supervised learning algorithm."
            }
            log.debug("Adding an unsupervised learning transform to the graph.")
            val node = LearningGraphNode.build(nextId++) {
                this.transform = transform
                sources += SubscribedTo(src, 0)
            }
            addNodeToGraph(node)
            return node
        }

        fun merge(vararg sources: GraphNode): GraphNode {
            val transform = MergeTransform.build {
                this.sources += sources
            }
            return addTransform(sources.asList(), transform)
        }

        /**
         * Adds a node to the graph that when run will produce a new data set by removing the given columns from its
         * input.
         */
        fun removeColumns(src: GraphNode, vararg toRemove: String): GraphNode {
            val transform = AllColumnsExceptTransform(toRemove.toSet())
            return addTransform(src, transform)
        }

        /**
         * Like the other [removeColumns] overload but takes [ColumnId] as input.
         */
        fun removeColumns(src: GraphNode, vararg toRemove: ColumnId<*>): GraphNode {
            return removeColumns(src, *toRemove.map { it.name }.toTypedArray())
        }

        /**
         * Adds a node to the graph that will return only a subsetColumns of the columns in it's input. Note that you can
         * also rename columns using this transform.
         */
        fun subsetColumns(src: GraphNode, init: ColumnSubsetTransform.Builder.() -> Unit): GraphNode {
            val transform = ColumnSubsetTransform.build {
                init()
            }
            return addTransform(src, transform)
        }

        /**
         * The inverse of [keepColumns], this drops all columns except those specified.
         */
        fun keepColumns(src: GraphNode, vararg column: ColumnId<*>): GraphNode {
            val transform = ColumnSubsetTransform.build {
                column.forEach {
                    keep(it)
                }
            }
            return addTransform(src, transform)
        }

        fun addTransform(sources: List<GraphNode>, transform: MultiTransform): GraphNode {
            return addTransform(sources, transform, nextId++)
        }

        /**
         * Add a [SupervisedLearningTransform] that gets its main data from `mainSource` and the supervised data from
         * `targetSource`. In other words, `mainSource` will provide the first parameter and `targetSource` will provide
         * the second parameter to [SupervisedLearningTransform.trainTransform].
         *
         * If the target comes from the same data set as the main data that is fine, just pass the same value for the
         * `mainSource` and `targetSource`.
         */
        fun addTransform(mainSource: GraphNode,
                targetSource: GraphNode,
                transform: SupervisedLearningTransform): GraphNode {
            val node = SupervisedLearningGraphNode.build(nextId++) {
                sources += SubscribedTo(mainSource, SupervisedLearningGraphNode.MAIN_DS_ID)
                if (targetSource == mainSource) {
                    log.debug("main source and target source are the same. " +
                            "Node {} will not have any trainOnlySources", this.id)
                } else {
                    trainOnlySources += SubscribedTo(targetSource, SupervisedLearningGraphNode.TARGET_DS_ID)
                }
                this.transform = transform
            }

            addNodeToGraph(node)
            return node
        }

        internal fun addTransform(sources: List<GraphNode>, transform: MultiTransform, nodeId: Int): GraphNode {
            val node = MultiTransformGraphNode.build(nodeId) {
                this.transform = transform
                this.sources += sources.mapIndexed { idx, source -> SubscribedTo(source, idx) }
            }

            addNodeToGraph(node)
            return node
        }

        internal fun addNodeToGraph(nodeToAdd: GraphNode) {
            check(!nodesById.containsKey(nodeToAdd.id))
            nodesById[nodeToAdd.id] = nodeToAdd
        }

        fun build(): DataGraph {
            return DataGraph(this)
        }

        /**
         * A Java-style builder for constucting a source definition.
         */
        inner class SourceBuilder(id: Int) {
            private val srcBuilder = SourceGraphNode.Builder(id)

            /**
             * Valid sources must include data with the type and name described by valId.
             */
            fun withValue(valId: ColumnId<*>): SourceBuilder {
                srcBuilder.columnIds += valId
                return this
            }

            /**
             * Valid sources must include data with the type and name described by the valIds.
             */
            fun withValues(vararg valIds: ColumnId<*>): SourceBuilder {
                srcBuilder.columnIds += valIds
                return this
            }

            /**
             * Valid sources must include data with the type and name described by valId during training. When just
             * getting predictions via the [DataGraph.transform] method this value is optional.
             */
            fun withTrainOnlyValue(valId: ColumnId<*>): SourceBuilder {
                srcBuilder.trainOnlyColumnIds += valId
                return this
            }

            /**
             * Valid sources must include data with the type and name described by these valIds during training. When
             * just getting predictions via the [DataGraph.transform] method this value is optional.
             */
            fun withTrainOnlyValues(vararg valIds: ColumnId<*>): SourceBuilder {
                srcBuilder.trainOnlyColumnIds += valIds
                return this
            }

            fun build(): GraphNode {
                val res = this@GraphBuilder.setSource(srcBuilder.build())
                return res
            }
        }
    }
}
