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
        tokens.associateBy { it.name }
    }

    private val tokenGroupMap: Map<String, ValueTokenGroup<*>> by lazy {
        tokenGroups.associateBy { it.name }
    }


    fun <T> token(valId: ValueId<T>): ValueToken<T> {
        val tok = tokenMap[valId.name] ?: throw IllegalArgumentException("Token ${valId.name} not found")
        if (tok.clazz == valId.clazz) {
            @Suppress("UNCHECKED_CAST")
            return tok as ValueToken<T>
        } else {
            throw IllegalArgumentException("Token ${valId.name} is not of type ${valId.clazz}")
        }
    }

    fun <T> token(name: String, clazz: Class<T>): ValueToken<T> {
        return token(ValueId(name, clazz))
    }

    inline fun <reified T : Any> token(name: String): ValueToken<T> {
        return token(name, T::class.java)
    }

    fun <T> tokenGroup(groupId: ValueIdGroup<T>): ValueTokenGroup<T> {
        if (groupId is AggregateValueIdGroup<T>) {
            return AggregateValueTokenGroup(groupId, this)
        } else {
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
    }

    fun <T> tokenGroup(name: String, clazz: Class<T>): ValueTokenGroup<T> {
        return tokenGroup(ValueIdGroup(name, clazz))
    }

    inline fun <reified T : Any> tokenGroup(name: String): ValueTokenGroup<T> {
        return tokenGroup(name, T::class.java)
    }
}
