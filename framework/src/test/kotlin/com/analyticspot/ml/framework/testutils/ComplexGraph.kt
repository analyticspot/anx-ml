package com.analyticspot.ml.framework.testutils

import com.analyticspot.ml.framework.datagraph.DataGraph
import com.analyticspot.ml.framework.datagraph.GraphNode
import com.analyticspot.ml.framework.description.ColumnId

/**
 * Functions that build big, complex [DataGraph] for testing.
 */

/**
 * See graph1.dot and graph1.png for the overall layout. This graph doesn't make much sense, but it is a complex graph.
 * In the png train-only links are dashed lines.
 *
 * This is a class so we can get access to some of the nodes and such.
 *
 * The overall graph here takes a data set containing words and a target. It then converts all the words to lowercase.
 * It then uses two supervised learning algorithms each of which learns to predict true if the word was seen with a
 * target of true. However, one of these is subscribed to an inverted target and one to a target that's been inverted
 * twice (i.e. not inverted). Finally we merge these and predict true iff both predict true. That can only happen
 * if the word appeared with both a "true" and a "false" target in the training data (as that way the word will have
 * appeared with a true inverted and a true non-inverted target).
 */
class Graph1 {
    val graph: DataGraph
    val targetId = ColumnId.create<Boolean>("target")
    val wordId = ColumnId.create<String>("word")
    val resultId = ColumnId.create<Boolean>("prediction")

    /**
     * The first Invert node. Handy to have a reference to it as it keeps track of how many times it was called.
     */
    val invert1: InvertBoolean
    val invert1Node: GraphNode

    /**
     * The second Invert node. Handy to have a reference to it as it keeps track of how many times it was called.
     */
    val invert2: InvertBoolean
    val invert2Node: GraphNode

    init {
        val bld = DataGraph.GraphBuilder()
        var src = bld.setSource {
            columnIds += wordId
            trainOnlyColumnIds += targetId
        }

        invert1 = InvertBoolean(targetId, targetId)
        invert1Node = bld.addTransform(src, invert1)

        invert2 = InvertBoolean(targetId, targetId)
        invert2Node = bld.addTransform(invert1Node, invert2)

        val lower = bld.addTransform(src, LowerCaseTransform(src.transformDescription))

        val trueIfNotSeenId = ColumnId.create<Boolean>("trueIfNotSeen")
        val trueIfNotSeen = bld.addTransform(lower, invert1Node,
                TrueIfSeenTransform(wordId, targetId, trueIfNotSeenId))

        val trueIfSeenId = ColumnId.create<Boolean>("trueIfSeen")
        val trueIfSeen = bld.addTransform(lower, invert2Node,
                TrueIfSeenTransform(wordId, targetId, trueIfSeenId))

        val merged = bld.merge(trueIfSeen, trueIfNotSeen)

        val andTrans = bld.addTransform(merged, AndTransform(listOf(trueIfNotSeenId, trueIfSeenId), resultId))
        bld.result = andTrans

        graph = bld.build()
    }

}
