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

/**
 * A [Subscription] indicates that a [GraphNode] requires the output of another [GraphNode]. Each [Subscription]
 * indicates the consuming node and an id. The source node then holds a reference to the subscription so that the
 * [GraphExecution] can notify subscribers when the data they require has been computed. The `subId` for a
 * [Subscription] is just an integer and is meaningful only to the subscriber. For example, a subscriber that has
 * subscribed to several data sets might assign each an index and use that as the `subId` so that when
 * [NodeExecutionManager] gets notified that some data is available it knows **which** data is now available.
 */
data class Subscription(val subscriber: GraphNode, val subId: Int)

/**
 * The other side of a [Subscription]: this indicates what node is subscribed to and what `subId` we expect when data
 * from that source is available.
 */
data class SubscribedTo(val source: GraphNode, val subId: Int)
