package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.datagraph.GraphNode
import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.dataset.IndirectDataSet
import com.analyticspot.ml.framework.description.IndirectValueToken
import com.analyticspot.ml.framework.description.ValueToken
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
    @JsonIgnore
    override val description: TransformDescription

    private val sources: List<GraphNode>

    init {
        sources = builder.sources

        val newTokens = mutableListOf<ValueToken<*>>()
        sources.forEachIndexed { idx, node ->
            node.tokens.forEach { newTokens += IndirectValueToken(idx, it) }
        }

        description = TransformDescription(
                tokens = newTokens,
                tokenGroups = sources.flatMap { it.tokenGroups }.toList())
    }

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

    override fun transform(dataSets: List<DataSet>): CompletableFuture<DataSet> =
            CompletableFuture.completedFuture(IndirectDataSet(dataSets))

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
