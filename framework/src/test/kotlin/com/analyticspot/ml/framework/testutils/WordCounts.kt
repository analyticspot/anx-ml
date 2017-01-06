package com.analyticspot.ml.framework.testutils

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.dataset.IterableDataSet
import com.analyticspot.ml.framework.datatransform.LearningTransform
import com.analyticspot.ml.framework.datatransform.TransformDescription
import com.analyticspot.ml.framework.description.IndexValueToken
import com.analyticspot.ml.framework.description.ValueId
import com.analyticspot.ml.framework.description.ValueIdGroup
import com.analyticspot.ml.framework.description.ValueToken
import com.analyticspot.ml.framework.description.ValueTokenGroupFromList
import com.analyticspot.ml.framework.observation.ArrayObservation
import com.analyticspot.ml.framework.observation.Observation
import com.fasterxml.jackson.annotation.JsonProperty
import com.fasterxml.jackson.annotation.JsonProperty.Access
import com.fasterxml.jackson.databind.annotation.JsonDeserialize
import com.fasterxml.jackson.databind.annotation.JsonPOJOBuilder
import org.slf4j.LoggerFactory
import java.util.concurrent.CompletableFuture

/**
 * An example of an unsupervised learner that doesn't know in advance how many outputs it will have. This one takes in
 * [Observation] instances that contain a `List<String>` and assigns each unique string to a output. The value of that
 * output is the number of times that string occurred. This is very much like a bag-of-words representation.
 *
 * @param sourceToken a [ValueToken] to retrieve the list of words from the input.
 * @param resultPrefix the prefix for the [ValueIdGroup]/[ValueTokenGroup] that this will produce. The resulting tokens
 * will be `prefix-X` where `X` is one of the words in the list.
 * @param wordMap map the word to the position of that word in the Observation array. (e.g. if wordMap["foo"]  = 7
 * then the count for the word "foo" will be at index 7 in the ArrayObservation that this produces when [transform] is
 * called. This is a constructor parameter so it can be deserialized. For training call the secondary constructor.
 */
@JsonDeserialize(builder = WordCounts.Builder::class)
class WordCounts private constructor(
        val sourceToken: ValueToken<List<String>>,
        resultId: ValueIdGroup<Int>,
        val wordMap: MutableMap<String, Int>) : LearningTransform {
    private val tokenGroupAndSetter = ValueTokenGroupFromList.create(resultId)

    @get:JsonProperty("resultId", access = Access.READ_ONLY)
    private val resultId: ValueIdGroup<Int>
        get() = tokenGroupAndSetter.tokenGroup.id

    override val description = TransformDescription(listOf(), tokenGroups = listOf(tokenGroupAndSetter.tokenGroup))

    constructor(sourceToken: ValueToken<List<String>>, resultId: ValueIdGroup<Int>)
            : this(sourceToken, resultId, mutableMapOf())

    companion object {
        private val log = LoggerFactory.getLogger(WordCounts::class.java)

        fun createFromSerialized(
                @JsonProperty("sourceToken") sourceToken: ValueToken<List<String>>,
                @JsonProperty("resultId") resultId: ValueIdGroup<Int>,
                @JsonProperty("wordMap") wordMap: MutableMap<String, Int>): WordCounts {
            return WordCounts(sourceToken, resultId, wordMap)
        }
    }

    override fun transform(dataSet: DataSet): CompletableFuture<DataSet> {
        val obsList = dataSet.map { transform(it) }
        return CompletableFuture.completedFuture(IterableDataSet(obsList))
    }

    fun transform(obs: Observation): Observation {
        val result = IntArray(wordMap.size)
        val words = obs.value(sourceToken)
        words.forEach {
            val idx = wordMap[it]
            if (idx == null) {
                log.debug("$it is not in the vocabulary. Ignored.")
            } else {
                result[idx] += 1
            }
        }
        // TODO: Converting from IntArray to Array<Int> is not very efficient as the latter is really a Java
        // Array<Integer> rather than an int[]. We should create Observation subclasses for primitive arrays.
        return ArrayObservation(result.toTypedArray())
    }

    override fun trainTransform(dataSet: DataSet): CompletableFuture<DataSet> {
        // Training phase
        dataSet.forEach { observation ->
            val words = observation.value(sourceToken)
            for (word in words) {
                if (!wordMap.containsKey(word)) {
                    wordMap[word] = wordMap.size
                }
            }
        }

        // Now that we know all the tokens, set them into the ValueTokenGroup
        val tokens = wordMap.map {
            val tokenName = "${tokenGroupAndSetter.tokenGroup.prefix}${ValueId.GROUP_SEPARATOR}${it.key}"
            IndexValueToken(it.value, ValueId.create<Int>(tokenName))
        }

        tokenGroupAndSetter.setter(tokens)
        return transform(dataSet)
    }

    // This should be unnecessary but is a workaround for this bug:
    //
    // https://github.com/FasterXML/jackson-databind/issues/1489
    //
    // We should be able to remove this when that gets fixed.
    @JsonPOJOBuilder(withPrefix = "set")
    class Builder() {
        lateinit var sourceToken: ValueToken<List<String>>
        lateinit var resultId: ValueIdGroup<Int>
        lateinit var wordMap: MutableMap<String, Int>

        fun build(): WordCounts = WordCounts(sourceToken, resultId, wordMap)
    }
}
