package com.analyticspot.ml.framework.datagraph

/**
 * A source node for a graph that is just a [DataSet] when nothing is known about the columns that will be in that
 * [DataSet]. See [SourceGraphNodeBase] for details.
 */
class DataSetSourceGraphNode(id: Int) : SourceGraphNodeBase(GraphNode.Builder(id))
