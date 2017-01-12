package com.analyticspot.ml.framework.description

/**
 * A [ValueIdGroup] that consists of multiple other [ValueId] and [ValueIdGroup].
 */
class AggregateValueIdGroup<ValueT>(builder: Builder<ValueT>) : ValueIdGroup<ValueT>(builder.name, builder.clazz) {
    internal val valueIds: List<ValueId<ValueT>> = builder.valueIds
    internal val valueIdGroups: List<ValueIdGroup<ValueT>> = builder.valueIdGroups

    companion object {
        inline fun <reified T : Any> builder(name: String): Builder<T> = Builder(name, T::class.java)

        inline fun <reified T : Any> build(name: String, init: Builder<T>.() -> Unit): AggregateValueIdGroup<T> {
            return with(Builder(name, T::class.java)) {
                init()
                build()
            }
        }
    }

    class Builder<ValueT>(val name: String, val clazz: Class<ValueT>) {
        val valueIds = mutableListOf<ValueId<ValueT>>()
        val valueIdGroups = mutableListOf<ValueIdGroup<ValueT>>()

        /**
         * Adds the given value id to the group and returns `this` for standard builder syntax.
         */
        fun withId(valId: ValueId<ValueT>): Builder<ValueT> {
            valueIds += valId
            return this
        }

        /**
         * Convenience overload of [withId] that allows you to specify just the name of the group (since the type, and
         * hence the id can be inferred).
         */
        fun withId(name: String): Builder<ValueT> {
            valueIds += ValueId(name, clazz)
            return this
        }

        /**
         * Adds a [ValueIdGroup]. See [withId].
         */
        fun withGroup(groupId: ValueIdGroup<ValueT>): Builder<ValueT> {
            valueIdGroups += groupId
            return this
        }

        /**
         * Convenience overload that allows you to specify just the name of the group. See [withId].
         */
        fun withGroup(name: String): Builder<ValueT> {
            valueIdGroups += ValueIdGroup(name, clazz)
            return this
        }

        fun build(): AggregateValueIdGroup<ValueT> = AggregateValueIdGroup(this)
    }
}
