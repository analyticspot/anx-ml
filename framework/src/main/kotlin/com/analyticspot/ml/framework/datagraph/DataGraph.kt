package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.dataset.SingleObservationDataSet
import com.analyticspot.ml.framework.datatransform.LearningTransform
import com.analyticspot.ml.framework.datatransform.MergeTransform
import com.analyticspot.ml.framework.datatransform.MultiTransform
import com.analyticspot.ml.framework.datatransform.SingleDataTransform
import com.analyticspot.ml.framework.datatransform.SupervisedLearningTransform
import com.analyticspot.ml.framework.observation.ArrayObservation
import com.analyticspot.ml.framework.observation.Observation
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
class DataGraph(builder: GraphBuilder) {
    val source: SourceGraphNode
    val result: GraphNode
    // An array of all the GraphNodes such that a node `x` can be found at `allNodes[x.id]`.
    internal val allNodes: Array<GraphNode>

    init {
        source = builder.source
        result = builder.result

        allNodes = Array<GraphNode>(builder.nodesById.size) {
            builder.nodesById[it] ?: throw IllegalStateException("No node in builder with id $it")
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
    }

    /**
     * Constructs an [Observation] that is compatible with the types/tokens specified for [source]. Note that this
     * will check that the values are compatible for either training or just tranforming. In other words type checking
     * will check trainOnly tokens if they're present and will ignore them if they're not.
     */
    fun buildSourceObservation(vararg values: Any): Observation {
        // Ensure that each value is valid (has the right type, etc.)
        values.forEachIndexed { idx, value ->
            check(value.javaClass == source.tokens[idx].clazz) {
                "Argument $idx (0-indexed) had type ${value.javaClass} but ${source.tokens[idx].clazz} was expected."
            }
        }
        val numTotalTokens = source.tokens.size
        val numTrainOnlyTokens = numTotalTokens - source.trainOnlyValueIds.size
        // Ensure that all values are present
        check(values.size == numTotalTokens || values.size == numTrainOnlyTokens) {
            "${values.size} values provided by $numTotalTokens required for training and $numTrainOnlyTokens " +
                    "required for transforming."
        }
        return ArrayObservation(values)
    }

    /**
     * Run the data through the entire graph. The result type is a future of `DataSet` because the graph
     * might contain an asynchronous [DataTransform] or it might contain an [OnDemandValue].
     */
    fun transform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        val graphExec = GraphExecution(this, ExecutionType.TRANSFORM, exec)
        return graphExec.execute(dataSet)
    }

    /**
     * Convenience overload that transforms a single [Observation].
     */
    fun transform(observation: Observation, exec: ExecutorService): CompletableFuture<Observation> {
        return transform(SingleObservationDataSet(observation), exec).thenApply {
            it.first()
        }
    }

    /**
     * Train the graph on the given inputs and return the result of transforming the source data set via the trained
     * transformers.
     */
    fun trainTransform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        val graphExec = GraphExecution(this, ExecutionType.TRAIN_TRANSFORM, exec)
        return graphExec.execute(dataSet)
    }

    class GraphBuilder {
        internal lateinit var source: SourceGraphNode

        lateinit var result: GraphNode

        // An array of GraphNode such that nodesById[idx] returns the GraphNode whose id is idx.
        internal val nodesById: MutableMap<Int, GraphNode> = mutableMapOf()

        internal var nextId = 0

        fun setSource(init: SourceGraphNode.Builder.() -> Unit): GraphNode {
            val sourceNode = SourceGraphNode.build(nextId++, init)
            return setSource(sourceNode)
        }

        internal fun setSource(node: SourceGraphNode): GraphNode {
            source = node
            check(!nodesById.containsKey(node.id))
            nodesById[node.id] = node
            return node
        }

        fun addTransform(src: GraphNode, transform: SingleDataTransform): GraphNode {
            return addTransform(src, transform, nextId++)
        }

        internal fun addTransform(src: GraphNode, transform: SingleDataTransform, nodeId: Int): GraphNode {
            log.debug("Adding an untrained transform to the graph.")
            val node = TransformGraphNode.build(nodeId) {
                this.transform = transform
                sources += SubscribedTo(src, 0)
            }
            src.subscribers += Subscription(node, 0)
            addNodeToGraph(node)
            return node
        }

        fun addTransform(src: GraphNode, transform: LearningTransform): GraphNode {
            log.debug("Adding an unsupervised learning transform to the graph.")
            val node = LearningGraphNode.build(nextId++) {
                this.transform = transform
                sources += SubscribedTo(src, 0)
            }
            src.subscribers += Subscription(node, 0)
            addNodeToGraph(node)
            return node
        }

        fun merge(vararg sources: GraphNode): GraphNode {
            val transform = MergeTransform.build {
                this.sources += sources
            }
            return addTransform(sources.asList(), transform)
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

            mainSource.subscribers += Subscription(node, SupervisedLearningGraphNode.MAIN_DS_ID)

            if (targetSource != mainSource) {
                targetSource.trainOnlySubscribers += Subscription(node, SupervisedLearningGraphNode.TARGET_DS_ID)
            }
            addNodeToGraph(node)
            return node
        }

        internal fun addTransform(sources: List<GraphNode>, transform: MultiTransform, nodeId: Int): GraphNode {
            val node = MultiTransformGraphNode.build(nodeId) {
                this.transform = transform
                this.sources += sources.mapIndexed { idx, source -> SubscribedTo(source, idx) }
            }
            node.sources.forEach { it.source.subscribers += Subscription(node, it.subId) }

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
    }
}
