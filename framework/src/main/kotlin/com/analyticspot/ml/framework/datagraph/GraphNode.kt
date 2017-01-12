package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.description.TransformDescription
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
    abstract val transformDescription: TransformDescription
    val tokens: List<ValueToken<*>>
        get() = transformDescription.tokens
    val tokenGroups: List<ValueTokenGroup<*>>
        get() = transformDescription.tokenGroups

    /**
     * Labels are used for injection during deserialization. See SERIALIZATION.README.md for details.
     */
    var label: String? = null

    private val tokenGroupMap: MutableMap<ValueIdGroup<*>, ValueTokenGroup<*>> = mutableMapOf()

    companion object {
        private val log = LoggerFactory.getLogger(GraphNode::class.java)
    }

    /**
     * Return a [NodeExecutionManager] for the given operation (`train`, `trainTransform`, or `execute`).
     */
    abstract fun getExecutionManager(parent: GraphExecution, execType: ExecutionType): NodeExecutionManager

    /**
     * Returns the token for this [ValueId].
     */
    fun <T> token(valId: ValueId<T>): ValueToken<T> {
        return transformDescription.token(valId)
    }

    /**
     * Convenience overload that is equivalent to `token(ValueId(name, clazz))`.
     */
    fun <T> token(name: String, clazz: Class<T>): ValueToken<T> {
        return transformDescription.token(name, clazz)
    }

    /**
     * Convenience overload in which the type information is determined automatically.
     */
    inline fun <reified T : Any> token(name: String): ValueToken<T> {
        return token(name, T::class.java)
    }

    /**
     * Returns the [ValueTokenGroup] that corresponds to the `groupId`. Note that this works with [ValueTokenGroup]
     * instances declared by this description in the [tokenGroups] member and with [AggregateValueTokenGroup] instances
     * constructed by users to group several [ValueId] and [ValueIdGroup] instances together.
     */
    fun <T> tokenGroup(groupId: ValueIdGroup<T>): ValueTokenGroup<T> {
        return transformDescription.tokenGroup(groupId)
    }

    /**
     * Convenience overload that is equivalent to `tokenGroup(ValueIdGroup(name, clazz))`.
     */
    fun <T> tokenGroup(name: String, clazz: Class<T>): ValueTokenGroup<T> {
        return transformDescription.tokenGroup(name, clazz)
    }

    /**
     * Convenience overload that determines type information from context.
     */
    inline fun <reified T : Any> tokenGroup(name: String): ValueTokenGroup<T> {
        return tokenGroup(name, T::class.java)
    }

    open class Builder(internal val id: Int) {
        val sources: MutableList<SubscribedTo> = mutableListOf()
        val trainOnlySources: MutableList<SubscribedTo> = mutableListOf()
    }
}

