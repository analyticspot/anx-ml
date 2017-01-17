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

package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.datagraph.GraphNode
import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.TransformDescription
import com.fasterxml.jackson.annotation.JacksonInject
import com.fasterxml.jackson.annotation.JsonIgnore
import com.fasterxml.jackson.annotation.JsonProperty
import com.fasterxml.jackson.annotation.JsonProperty.Access
import com.fasterxml.jackson.databind.annotation.JsonDeserialize
import com.fasterxml.jackson.databind.annotation.JsonPOJOBuilder
import java.util.concurrent.CompletableFuture

/**
 * A [MultiTransform] that takes in multiple [DataSet] instances and concatenates them into a single output.
 */
@JsonDeserialize(builder = MergeTransform.DeserBuilder::class)
class MergeTransform (builder: Builder) : MultiTransform {
    @get:JsonIgnore
    override val description: TransformDescription by lazy {
        val allColumns = builder.sources.flatMap { it.transformDescription.columns }
        val allColGroups = builder.sources.flatMap { it.transformDescription.columnGroups }
        TransformDescription(allColumns, allColGroups)
    }

    private val sources: List<GraphNode> = builder.sources

    @get:JsonProperty(access = Access.READ_ONLY)
    private val sourceIds: List<Int> by lazy {
        sources.map { it.id }
    }

    companion object {
        fun build(init: Builder.() -> Unit): MergeTransform {
            return with(Builder()) {
                init()
                build()
            }
        }
    }

    override fun transform(dataSets: List<DataSet>): CompletableFuture<DataSet> {
        val resultDs = DataSet.build {
            for (dataSet in dataSets) {
                addAll(dataSet)
            }
        }
        return CompletableFuture.completedFuture(resultDs)
    }

    open class Builder {
        val sources = mutableListOf<GraphNode>()

        fun build(): MergeTransform = MergeTransform(this)
    }

    /**
     * A builder used specifically when we deserialize one of these from JSON. It takes an array of [GraphNode] in its
     * constructor. These nodes are used to generate [ValueToken] for the inputs.
     */
    @JsonPOJOBuilder(withPrefix = "set")
    class DeserBuilder(
            @JacksonInject(MultiTransform.JSON_SOURCE_INJECTION_ID) private val injectedSources: List<GraphNode>)
        : Builder() {
        fun setSourceIds(ids: List<Int>) {
            // Called by Jackson with the list of source ids. We use this to double check that the Jackson injected list
            // of sources is in the correct order.
            val injectedIds = injectedSources.map { it.id }.toList()
            check(injectedIds.equals(ids)) {
                "Expected data sources in this order ${ids} but they were injected in this order: $injectedIds"
            }
            sources += injectedSources
        }
    }
}
