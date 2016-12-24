package com.analyticspot.ml.framework.description

import com.analyticspot.ml.framework.datatransform.DataTransform
import java.util.SortedMap

/**
 * A description of a [DataSet]. This doesn't contain any data but it does know how to generate data of the given type,
 * including knowing all the other [DataSet] and [DataTransform] inputs required to realize this data set. Thus, a
 * single [DataDescription] is sufficient to reproduce the entire data graph that terminates with this description.
 */
abstract class DataDescription(val tokens: List<ValueToken<*>>, val tokenGroups: List<ValueTokenGroup<*>>) {
    private val tokenMap: SortedMap<String, ValueToken<*>> = sortedMapOf()
    private val tokenGroupMap: SortedMap<String, ValueTokenGroup<*>> = sortedMapOf()

    init {
        tokens.forEach {
            check(!tokenMap.containsKey(it.name)) {
                "A token with name ${it.name} is already present in this data set."
            }
            tokenMap.put(it.name, it)
        }

        tokenGroups.forEach {
            check(!tokenGroupMap.containsKey(it.prefix)) {
                "A token group with prefix ${it.prefix} is already present in this data set."
            }
            tokenGroupMap.put(it.prefix, it)
        }

    }

    fun <T> token(name: String, clazz: Class<T>): ValueToken<T> {
        val token = tokenMap[name] ?: throw IllegalArgumentException("Token $name not found")
        if (token.clazz == clazz) {
            @Suppress("UNCHECKED_CAST")
            return token as ValueToken<T>
        } else {
            throw IllegalArgumentException("Token $name is not of type $clazz")
        }
    }

    inline fun <reified T : Any> token(name: String): ValueToken<T> {
        return token(name, T::class.java)
    }

    fun <T> tokenGroup(prefix: String, clazz: Class<T>): ValueTokenGroup<T> {
        val tokenGroup = tokenGroupMap[prefix]
                ?: throw IllegalArgumentException("TokenGroup with prefix $prefix not found")
        if (tokenGroup.clazz == clazz) {
            @Suppress("UNCHECKED_CAST")
            return tokenGroup as ValueTokenGroup<T>
        } else {
            throw IllegalArgumentException("TokenGroup with prefix $prefix is not of type $clazz")
        }
    }

    inline fun <reified T : Any> tokenGroup(name: String): ValueTokenGroup<T> {
        return tokenGroup(name, T::class.java)
    }
}
