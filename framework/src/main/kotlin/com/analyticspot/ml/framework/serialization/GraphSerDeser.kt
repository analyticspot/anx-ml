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

import com.analyticspot.ml.framework.datagraph.DataGraph
import com.analyticspot.ml.framework.datagraph.DataSetSourceGraphNode
import com.analyticspot.ml.framework.datagraph.GraphNode
import com.analyticspot.ml.framework.datagraph.HasTransformGraphNode
import com.analyticspot.ml.framework.datagraph.SourceGraphNode
import com.analyticspot.ml.framework.datagraph.sort
import com.analyticspot.ml.framework.datatransform.DataTransform
import com.analyticspot.ml.framework.datatransform.MultiTransform
import com.analyticspot.ml.framework.datatransform.SingleDataTransform
import com.analyticspot.ml.framework.description.ColumnId
import com.fasterxml.jackson.annotation.JsonInclude
import com.fasterxml.jackson.annotation.JsonInclude.Include
import com.fasterxml.jackson.annotation.JsonTypeInfo
import com.fasterxml.jackson.annotation.JsonTypeInfo.As
import com.fasterxml.jackson.annotation.JsonTypeInfo.Id
import com.fasterxml.jackson.databind.annotation.JsonDeserialize
import com.fasterxml.jackson.databind.annotation.JsonPOJOBuilder
import org.slf4j.LoggerFactory
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream

/**
 * Serializes and deserializes entire [DataGraph] instances. See `SERIALIZATION.README.md` for details.
 */
class GraphSerDeser {
    private val labelToDeserializer = mutableMapOf<String, TransformFactory<*>>()
    private val formatMap = mutableMapOf<Class<out Format<*>>, Format<*>>(
            StandardJsonFormat::class.java to StandardJsonFormat()
    )

    companion object {
        private val log = LoggerFactory.getLogger(GraphSerDeser::class.java)

        const val MAIN_GRAPH_FILENAME = "graph.json"
        const val SOURCE_NAME = "source"
        const val RESULT_NAME = "result"
    }

    /**
     * Serializes `graph` to the given `output`. Note that **this** will convert `output` into a `ZipOutputStream` so
     * callers should pass just a plain `FileOutputStream` or similar.
     *
     * This will close the passed OutputStream.
     */
    fun serialize(graph: DataGraph, output: OutputStream) {
        val zipOut = ZipOutputStream(output)

        // Write graph.json
        val graphJsonEntry = ZipEntry(MAIN_GRAPH_FILENAME)
        zipOut.putNextEntry(graphJsonEntry)

        val topoOrder = sort(graph)
        val outObj = GraphStucture(graph.source.id, graph.result.id, graph.metaData)
        topoOrder.forEach {
            val serNode: SerGraphNode = when (it) {
                is SourceGraphNode -> SourceSerGraphNode.create(it)
                is DataSetSourceGraphNode -> DataSetSourceSerGraphNode.create(it)
                // Note that we only serialize what's necessary to apply a model, not train one so we can treat
                // learning and non-learning transforms the same way.
                is HasTransformGraphNode<*> -> {
                    val format = formatMap[it.transform.formatClass] ?:
                            throw IllegalStateException("Unknown format: ${it.transform.formatClass}")
                    TransformSerGraphNode.create(it, format.getMetaData(it.transform))
                }
                else -> throw IllegalStateException("Unknown GraphNode type:  ${it.javaClass.canonicalName}")
            }
            outObj.graph.add(serNode)
        }

        JsonMapper.mapper.writeValue(zipOut, outObj)

        // close graph.json
        zipOut.closeEntry()

        // Now write one more file for each node in the graph that's a TransformGraphNode
        topoOrder.forEach {
            when (it) {
                is HasTransformGraphNode<*> -> {
                    val nodeEntry = ZipEntry(it.id.toString())
                    zipOut.putNextEntry(nodeEntry)
                    serializeTransform(it.transform, zipOut)
                    zipOut.closeEntry()
                }
            }
        }

        zipOut.close()
    }

    /**
     * Convenience overload that creates `file`, writes to it, and closes it.
     */
    fun serialize(graph: DataGraph, file: File) {
        val fileOut = FileOutputStream(file)
        serialize(graph, fileOut)
    }

    /**
     * Convenience overload which creates a file with the given path and then serializes the [DataGraph] to it.
     */
    fun serialize(graph: DataGraph, filePath: String) {
        val f = File(filePath)
        serialize(graph, f)
    }

    /**
     * Deserialize a .zip file as written by one of the [serialize] methods back to a [DataGraph]. Note that the
     * `input` should just be a regular `InputStream` (e.g. a `FileInputStream`); this will take care of wrapping it
     * in a `ZipInputStream`.
     */
    fun deserialize(input: InputStream): DataGraph {
        // Note: the serialize method should guarantee that (1) graph.json is the first entry in the stream and
        // (2) that the files in the stream for each node will be in topological order.

        // Map from id of node to the data for that node.
        val deserializedNodeMap = mutableMapOf<Int, GraphNode>()

        val zipIn = ZipInputStream(input)
        var zipEntry: ZipEntry? = zipIn.nextEntry
        check(zipEntry != null && zipEntry.name == MAIN_GRAPH_FILENAME) {
            "$MAIN_GRAPH_FILENAME must be the first file in the zip"
        }
        val graphData = JsonMapper.mapper.readValue(zipIn, GraphStucture::class.java)
        zipIn.closeEntry()

        val graphBuilder = DataGraph.GraphBuilder()
        graphBuilder.metaData = graphData.metaData

        for (graphDataNode in graphData.graph) {
            val nodeId = graphDataNode.id
            var newNode: GraphNode? = null
            log.debug("Deserializing node {}", nodeId)
            when (graphDataNode) {
                is TransformSerGraphNode -> {
                    val nodeSources = graphDataNode.sources.map {
                        deserializedNodeMap[it] ?: throw IllegalStateException("Unable to find node source $it")
                    }

                    zipEntry = zipIn.nextEntry ?: throw IllegalStateException("No data for node $nodeId")
                    val transform = deserializeTransform(
                            graphDataNode.label, graphDataNode.metaData, nodeSources, zipIn)
                    zipIn.closeEntry()

                    when (transform) {
                        is SingleDataTransform -> {
                            check(nodeSources.size == 1)
                            newNode = graphBuilder.addTransform(nodeSources[0], transform, nodeId)
                        }

                        is MultiTransform -> {
                            newNode = graphBuilder.addTransform(nodeSources, transform, nodeId)
                        }

                        else -> throw IllegalStateException("Unknown transform type " + transform.javaClass)
                    }
                    check(newNode.id == zipEntry.name.toInt())
                }

                is SourceSerGraphNode -> {
                    check(nodeId == graphData.sourceId)
                    val sourceNode = SourceGraphNode.build(graphData.sourceId) {
                        columnIds += graphDataNode.columnIds.minus(graphDataNode.trainOnlyColumnIds)
                        trainOnlyColumnIds += graphDataNode.trainOnlyColumnIds
                    }
                    newNode = graphBuilder.setSource(sourceNode)
                }

                is DataSetSourceSerGraphNode -> {
                    check(nodeId == graphData.sourceId)
                    newNode = graphBuilder.setDataSetSource(DataSetSourceGraphNode(nodeId))
                }
            }

            deserializedNodeMap[newNode!!.id] = newNode
        }
        graphBuilder.result = deserializedNodeMap[graphData.resultId] ?:
                throw IllegalStateException("Result node not found.")

        graphBuilder.missingTrainNodes = true
        return graphBuilder.build()
    }

    /**
     * Indicates that the given factory should be used for any [GraphNode] with the given label instead of using the
     * format specified by the [DataTransform].
     */
    fun registerFactoryForLabel(label: String, factory: TransformFactory<*>) {
        check(!labelToDeserializer.containsKey(label)) {
            "There is already factory registered for label $label"
        }
        labelToDeserializer[label] = factory
    }

    /**
     * Register an additional [Format] with the [GraphSerDeser]. Note that [StandardJsonFormat] is always registred and
     * need not be added.
     */
    fun registerFormat(format: Format<*>) {
        val clazz = format.javaClass
        check(!formatMap.containsKey(clazz)) {
            "There is already a format registered with type $clazz"
        }
        formatMap[clazz] = format
    }

    // Serialize a single DataTransform.
    // Visible for testing
    internal fun serializeTransform(transform: DataTransform, output: OutputStream) {
        val format = formatMap[transform.formatClass] ?:
                throw IllegalStateException("${transform.formatClass} is not registered")
        format.serialize(transform, output)
    }

    // Deserialize a single DataTransform.
    // Visible for testing
    internal fun <MetaDataT : FormatMetaData> deserializeTransform(label: String?, metaData: MetaDataT,
            sources: List<GraphNode>, input: InputStream): DataTransform {
        val factoryToUse: TransformFactory<MetaDataT>
        if (label != null && labelToDeserializer.containsKey(label)) {
            log.info("Using custom TransformFactory for label {}", label)
            val factory = labelToDeserializer[label] ?: throw IllegalStateException("Should be impossible")
            @Suppress("UNCHECKED_CAST")
            factoryToUse = factory as TransformFactory<MetaDataT>
        } else {
            val format = formatMap[metaData.formatClass] ?:
                    throw IllegalStateException("Unknown format ${metaData.formatClass}")
            if (format.metaDataClass == metaData.javaClass) {
                @Suppress("UNCHECKED_CAST")
                factoryToUse = format as Format<MetaDataT>
            } else {
                throw IllegalStateException("Format ${format.javaClass} expects metadata of type " +
                        "${format.metaDataClass} but found meta data of type ${metaData.javaClass}")
            }
        }
        return factoryToUse.deserialize(metaData, sources, input)
    }

    // We really just want to serialize a Map<Int, SerGraphNode> but that doesn't work quite right due to type erasure.
    // Specifically, Jackson knows it's a Map, but not the types in the map. As a result it can't find the @JsonTypeInfo
    // annotation on the base class and the types aren't serialized. In general, Jackson recommends that you don't
    // serialize Map, List, etc. as the "root type": https://github.com/FasterXML/jackson-databind/issues/336
    //
    // Also, we want to serialize the source and result node id's and you can't do that in just a Map 'cause the
    // value type is different. So we use a tiny wrapper class here to make everything work.
    //
    // `topoOrder` is a list of graph ids in a valid topological order. We need this because the graph is otherwise
    // just ordered by node id.
    private class GraphStucture(val sourceId: Int, val resultId: Int, val metaData: String?) {
        val graph = mutableListOf<SerGraphNode>()
    }

    // The following class hierarchy is the serialization format for GraphNodes. It's quite different from the
    // GraphNode's themselves so we simply create new classes with the appropriate getters and setters which
    // return the properties we want to serialize and in the way we want them serialized.
    @JsonTypeInfo(use=Id.CLASS, include=As.PROPERTY, property="class")
    open class SerGraphNode(builder: Builder) {
        @JsonInclude(Include.NON_NULL)
        val label: String?
        @JsonInclude(Include.NON_EMPTY)
        val subscribers: List<Int>
        @JsonInclude(Include.NON_EMPTY)
        val sources: List<Int>
        val id: Int

        init {
            label = builder.label
            subscribers = builder.subscribers
            sources = builder.sources
            id = builder.id ?: throw IllegalArgumentException("Id can not be null")

        }

        open class Builder {
            var sources: List<Int> = listOf()
            var subscribers: List<Int> = listOf()
            var label: String? = null
            var id: Int? = null

            fun fromNode(node: GraphNode) {
                subscribers = node.subscribers.map { it.subscriber.id }
                sources = node.sources.map { it.source.id }
                id = node.id
                label = node.label
            }
        }
    }

    @JsonDeserialize(builder = DataSetSourceSerGraphNode.Builder::class)
    class DataSetSourceSerGraphNode(builder: Builder) : SerGraphNode(builder) {
        companion object {
            fun create(node: DataSetSourceGraphNode): DataSetSourceSerGraphNode {
                val bldr = Builder()
                bldr.fromNode(node)
                return DataSetSourceSerGraphNode(bldr)
            }
        }

        @JsonPOJOBuilder(withPrefix = "set")
        class Builder : SerGraphNode.Builder() {
            fun build(): DataSetSourceSerGraphNode = DataSetSourceSerGraphNode(this)
        }
    }

    @JsonDeserialize(builder = SourceSerGraphNode.Builder::class)
    class SourceSerGraphNode(builder: Builder) : SerGraphNode(builder) {
        val columnIds: List<ColumnId<*>>
        @JsonInclude(Include.NON_EMPTY)
        val trainOnlyColumnIds: List<ColumnId<*>>

        init {
            columnIds = builder.columnIds
            trainOnlyColumnIds = builder.trainOnlyColumnIds
        }

        companion object {
            fun create(node: SourceGraphNode): SourceSerGraphNode {
                return with(Builder()) {
                    fromNode(node)
                    build()
                }
            }
        }

        @JsonPOJOBuilder(withPrefix = "set")
        class Builder : SerGraphNode.Builder() {
            var columnIds: List<ColumnId<*>> = listOf()
            var trainOnlyColumnIds: List<ColumnId<*>> = listOf()

            fun fromNode(node: SourceGraphNode) {
                columnIds = node.columnIds
                trainOnlyColumnIds += node.trainOnlyColumnIds
                super.fromNode(node)
            }

            fun build(): SourceSerGraphNode {
                return SourceSerGraphNode(this)
            }
        }
    }

    @JsonDeserialize(builder = TransformSerGraphNode.Builder::class)
    class TransformSerGraphNode(builder: Builder) : SerGraphNode(builder) {
        val metaData: FormatMetaData

        companion object {
            internal fun create(node: HasTransformGraphNode<*>, metaData: FormatMetaData): TransformSerGraphNode {
                return with(Builder()) {
                    fromNode(node)
                    this.metaData = metaData
                    build()
                }
            }
        }

        init {
            metaData = builder.metaData ?: throw IllegalArgumentException("Missing FormatMetaData")
        }

        @JsonPOJOBuilder(withPrefix = "set")
        class Builder : SerGraphNode.Builder() {
            var metaData: FormatMetaData? = null

            fun build(): TransformSerGraphNode {
                return TransformSerGraphNode(this)
            }
        }
    }
}
