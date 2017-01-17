/*
 * Copyright (C) 2017 Analytic Spot.
 * 
 * This file is part of the ANX ML library.
 * 
 * The ANX ML library is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser
 * General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 * 
 * Foobar is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License along with the ANX ML libarary.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.description.ColumnIdGroup
import com.analyticspot.ml.framework.description.TransformDescription
import org.slf4j.LoggerFactory

/**
 * This is the base class for all [GraphNode]s. Each such node represents a single node in the graph. It holds the
 * metadata about that node (what its inputs are, what its output is, how it transforms its input into its output,
 * etc.).
 */
abstract class GraphNode internal constructor(builder: Builder) {
    internal var sources: List<SubscribedTo> = builder.sources
    internal var trainOnlySources: List<SubscribedTo> = builder.trainOnlySources
    internal val subscribers: MutableList<Subscription> = mutableListOf()
    internal val trainOnlySubscribers: MutableList<Subscription> = mutableListOf()
    internal val id: Int = builder.id
    abstract val transformDescription: TransformDescription
    val columns: List<ColumnId<*>>
        get() = transformDescription.columns
    val columnGroups: List<ColumnIdGroup<*>>
        get() = transformDescription.columnGroups

    /**
     * Labels are used for injection during deserialization. See SERIALIZATION.README.md for details.
     */
    var label: String? = null

    companion object {
        private val log = LoggerFactory.getLogger(GraphNode::class.java)
    }

    /**
     * Return a [NodeExecutionManager] for the given operation (`train`, `trainTransform`, or `execute`).
     */
    abstract fun getExecutionManager(parent: GraphExecution, execType: ExecutionType): NodeExecutionManager

    open class Builder(internal val id: Int) {
        val sources: MutableList<SubscribedTo> = mutableListOf()
        val trainOnlySources: MutableList<SubscribedTo> = mutableListOf()
    }
}

