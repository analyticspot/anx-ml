package com.analyticspot.ml.framework.description

/**
 * Describes the outputs produced by an execution of the node. The main purpose here is to allow each [DataTransform] to
 * specify the [ValueToken] types that will be most efficient for the data structure they produce. Note that this data
 * is available **before** training. Thus this can only tell you about tokens and token groups that the transform knows
 * it will produce. However, for transforms that produce an unknown number of outputs there will be token groups but
 * the actual tokens will be available only via those groups.
 */
class TransformDescription(val tokens: List<ValueToken<*>>,
        val tokenGroups: List<ValueTokenGroup<*>> = listOf()) {

    private val tokenMap: Map<String, ValueToken<*>> by lazy {
        tokens.plus(tokenGroups.flatMap { it.declaredTokens }).associateBy { it.name }
    }

    private val tokenGroupMap: Map<String, ValueTokenGroup<*>> by lazy {
        tokenGroups.associateBy { it.name }
    }

    /**
     * Returns the token that corresponds to the [ValueId].
     */
    fun <T> token(valId: ValueId<T>): ValueToken<T> {
        val tok = tokenMap[valId.name] ?: throw IllegalArgumentException("Token ${valId.name} not found")
        if (tok.clazz == valId.clazz) {
            @Suppress("UNCHECKED_CAST")
            return tok as ValueToken<T>
        } else {
            throw IllegalArgumentException("Token ${valId.name} is not of type ${valId.clazz}")
        }
    }

    /**
     * Convenience overload of [token] that is equivalent to `token(ValueId(name, clazz))`.
     */
    fun <T> token(name: String, clazz: Class<T>): ValueToken<T> {
        return token(ValueId(name, clazz))
    }

    /**
     * Convenience overload of [token] that infers the type information from the context.
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
        val tokGroup = tokenGroupMap[groupId.name] ?:
                throw IllegalArgumentException("No token group found with id $groupId")
        if (tokGroup.clazz == groupId.clazz) {
            @Suppress("UNCHECKED_CAST")
            return tokGroup as ValueTokenGroup<T>
        } else {
            throw IllegalArgumentException(
                    "TokenGroup $groupId has type ${tokGroup.clazz} but ${groupId.clazz} was passed with the id"
            )
        }
    }

    /**
     * Convenience overload of [tokenGroup] that is equivalent to `tokenGroup(TokenGroupId(name, clazz))`.
     */
    fun <T> tokenGroup(name: String, clazz: Class<T>): ValueTokenGroup<T> {
        return tokenGroup(ValueIdGroup(name, clazz))
    }

    /**
     * Convenience overload of [tokenGroup] that infers the type information from the context.
     */
    inline fun <reified T : Any> tokenGroup(name: String): ValueTokenGroup<T> {
        return tokenGroup(name, T::class.java)
    }
}
