package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.datatransform.DataTransform
import com.analyticspot.ml.framework.observation.Observation
import org.slf4j.LoggerFactory
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 *
 */
internal open class TransformGraphNode protected constructor(builder: Builder) : GraphNode(builder) {
    val transform: DataTransform = builder.transform ?: throw IllegalArgumentException("Transform can not be null")

    companion object {
        private val log = LoggerFactory.getLogger(Companion::class.java)
        fun build(id: Int, init: Builder.() -> Unit): TransformGraphNode {
            return with(Builder(id)) {
                init()
                build()
            }
        }
    }

    override fun transformWithSource(graphSource: Observation, exec: ExecutorService): CompletableFuture<Observation> {
        log.info("transformWithSource called on a TransformGraphNode. Computing source.")
        check(sources.size == 1)
        // Note: we should be able to just do "thenApplyAsync({transform.transform(it)}, exec) but that fails. It
        // appears to be a bug in Kotlin. See https://youtrack.jetbrains.com/issue/KT-15432.
        val result = CompletableFuture<Observation>()
        sources[0].transformWithSource(graphSource, exec).thenApply {
            log.debug("Source computation complete. Scheduling computation of this value.")
            exec.submit {
                val resultObs = transform.transform(it)
                log.debug("Done computing my own value. Completing the future.")
                result.complete(resultObs)
            }
        }
        return result
    }

    open class Builder(id: Int) : GraphNode.Builder(id) {
        var transform: DataTransform? = null
            set(value) {
                field = value ?: throw IllegalArgumentException("Transform can not be null")
                tokens.addAll(value.description.tokens)
                tokenGroups.addAll(value.description.tokenGroups)
            }

        override fun build(): TransformGraphNode = TransformGraphNode(this)
    }
}

