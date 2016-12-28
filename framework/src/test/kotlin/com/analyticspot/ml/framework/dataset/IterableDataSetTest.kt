package com.analyticspot.ml.framework.dataset

import com.analyticspot.ml.framework.description.IndexValueToken
import com.analyticspot.ml.framework.description.ValueId
import com.analyticspot.ml.framework.observation.ArrayObservation
import com.analyticspot.ml.framework.observation.equalValues
import org.assertj.core.api.Assertions.assertThat
import org.slf4j.LoggerFactory
import org.testng.annotations.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class IterableDataSetTest {
    companion object {
        private val log = LoggerFactory.getLogger(IterableDataSetTest::class.java)
    }

    @Test
    fun testSingleIterationWorks() {
        val srcData = listOf(
                ArrayObservation.create("hi", 10),
                ArrayObservation.create("there", 20),
                ArrayObservation.create("friends", 22))
        val ds = IterableDataSet(srcData)

        val tokens = listOf(
                IndexValueToken.create(0, ValueId.create<String>("v1")),
                IndexValueToken.create(1, ValueId.create<Int>("v2"))
        )

        val iter = ds.iterator()
        assertThat(iter.hasNext()).isTrue()
        val v1 = iter.next()
        assertThat(equalValues(tokens, v1, srcData[0])).isTrue()

        assertThat(iter.hasNext()).isTrue()
        val v2 = iter.next()
        assertThat(equalValues(tokens, v2, srcData[1])).isTrue()

        assertThat(iter.hasNext()).isTrue()
        val v3 = iter.next()
        assertThat(equalValues(tokens, v3, srcData[2])).isTrue()

        assertThat(iter.hasNext()).isFalse()
    }

    // Since a DataSet can be passed to multiple GraphNodes it has to be "restartable" so here we ensure that we can
    // iterate over the same data set more than once.
    @Test
    fun testMultipleIterationsWork() {
        val srcData = listOf(
                ArrayObservation.create("hi", 10),
                ArrayObservation.create("there", 20),
                ArrayObservation.create("friends", 22))
        val ds = IterableDataSet(srcData)

        val tokens = listOf(
                IndexValueToken.create(0, ValueId.create<String>("v1")),
                IndexValueToken.create(1, ValueId.create<Int>("v2"))
        )

        fun checkIteration() {
            val iter = ds.iterator()
            assertThat(iter.hasNext()).isTrue()
            val v1 = iter.next()
            assertThat(equalValues(tokens, v1, srcData[0])).isTrue()

            assertThat(iter.hasNext()).isTrue()
            val v2 = iter.next()
            assertThat(equalValues(tokens, v2, srcData[1])).isTrue()

            assertThat(iter.hasNext()).isTrue()
            val v3 = iter.next()
            assertThat(equalValues(tokens, v3, srcData[2])).isTrue()

            assertThat(iter.hasNext()).isFalse()
        }
        checkIteration()
        checkIteration()
        checkIteration()
    }

    // Like the above but we also ensure that iteration is thread safe.
    @Test
    fun testIterationsIsThreadSafe() {
        val srcData = listOf(
                ArrayObservation.create("hi", 10),
                ArrayObservation.create("there", 20),
                ArrayObservation.create("friends", 22))
        val ds = IterableDataSet(srcData)

        val tokens = listOf(
                IndexValueToken.create(0, ValueId.create<String>("v1")),
                IndexValueToken.create(1, ValueId.create<Int>("v2"))
        )

        fun checkIteration() {
            val iter = ds.iterator()
            assertThat(iter.hasNext()).isTrue()
            val v1 = iter.next()
            assertThat(equalValues(tokens, v1, srcData[0])).isTrue()

            assertThat(iter.hasNext()).isTrue()
            val v2 = iter.next()
            assertThat(equalValues(tokens, v2, srcData[1])).isTrue()

            assertThat(iter.hasNext()).isTrue()
            val v3 = iter.next()
            assertThat(equalValues(tokens, v3, srcData[2])).isTrue()

            assertThat(iter.hasNext()).isFalse()
        }

        // Use a CountdownLatch to start all threads at once to try to maximize chances of race conditions.
        val numThreads = 4
        val startLatch = CountDownLatch(numThreads)
        val errorsInThreads = AtomicBoolean(false)
        val exec = Executors.newFixedThreadPool(numThreads)
        for (i in 1..numThreads) {
            exec.submit {
                startLatch.countDown()
                startLatch.await()
                try {
                    checkIteration()
                } catch (t: Throwable) {
                    log.error("Error in thread:", t)
                    errorsInThreads.set(true)
                }
            }
        }
        exec.shutdown()
        exec.awaitTermination(10, TimeUnit.MINUTES)
        assertThat(errorsInThreads.get()).isFalse()
    }
}
