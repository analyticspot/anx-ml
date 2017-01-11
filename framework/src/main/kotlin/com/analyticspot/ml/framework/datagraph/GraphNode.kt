package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.datatransform.TransformDescription
import com.analyticspot.ml.framework.description.AggregateValueIdGroup
import com.analyticspot.ml.framework.description.AggregateValueTokenGroup
import com.analyticspot.ml.framework.description.ValueId
import com.analyticspot.ml.framework.description.ValueIdGroup
import com.analyticspot.ml.framework.description.ValueToken
import com.analyticspot.ml.framework.description.ValueTokenGroup
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
    val transformDescription = builder.transformDescription
    val tokens: List<ValueToken<*>> = transformDescription.tokens
    /**
     * These are the [ValueTokenGroup] instances produced by the transform. However, you can always create additional
     * [ValueTokenGroup] instances for custom groupings of [ValueToken] and [ValueTokenGroup] via ...
     */
    val tokenGroups: List<ValueTokenGroup<*>>

    /**
     * Labels are used for injection during deserialization. See SERIALIZATION.README.md for details.
     */
    var label: String? = null

    private val tokenGroupMap: MutableMap<ValueIdGroup<*>, ValueTokenGroup<*>> = mutableMapOf()

    init {
        tokenGroups = builder.tokenGroups

        tokenGroups.forEach {
            tokenGroupMap[it.id] = it
        }
    }

    companion object {
        private val log = LoggerFactory.getLogger(GraphNode::class.java)
    }

    /**
     * Return a [NodeExecutionManager] for the given operation (`train`, `trainTransform`, or `execute`).
     */
    abstract fun getExecutionManager(parent: GraphExecution, execType: ExecutionType): NodeExecutionManager

    fun <T> token(valId: ValueId<T>): ValueToken<T> {
        return transformDescription.token(valId)
    }


    fun <T> tokenGroup(groupId: ValueIdGroup<T>): ValueTokenGroup<T> {
        if (groupId is AggregateValueIdGroup<T>) {
            return AggregateValueTokenGroup(groupId, this)
        } else {
            val tokGroup = tokenGroupMap[groupId] ?:
                    throw IllegalArgumentException("No token group found with id $groupId")
            if (tokGroup.clazz == groupId.clazz) {
                @Suppress("UNCHECKED_CAST")
                return tokGroup as ValueTokenGroup<T>
            } else {
                throw IllegalArgumentException(
                        "TokenGroup $groupId has type ${tokGroup.clazz} but ${groupId.clazz} was passed wiht the id"
                )
            }
        }
    }

    open class Builder(internal val id: Int) {
        lateinit var transformDescription: TransformDescription
        val tokenGroups: MutableList<ValueTokenGroup<*>> = mutableListOf()
        val sources: MutableList<SubscribedTo> = mutableListOf()
        val trainOnlySources: MutableList<SubscribedTo> = mutableListOf()
    }
}

