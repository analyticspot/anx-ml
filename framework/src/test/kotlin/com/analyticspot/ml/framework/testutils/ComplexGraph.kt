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

        val lower = bld.addTransform(src, LowerCaseTransform())

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
