package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.description.ValueId
import com.analyticspot.ml.framework.description.ValueToken
import com.analyticspot.ml.framework.description.ValueTokenGroup

/**
 * Describes the outputs produced by an execution of the node. The main purpose here is to allow each [DataTransform] to
 * specify the [ValueToken] types that will be most efficient for the data structure they produce.
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
}
