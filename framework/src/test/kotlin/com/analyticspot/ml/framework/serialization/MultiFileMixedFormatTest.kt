package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datagraph.DataGraph
import com.analyticspot.ml.framework.datagraph.HasTransformGraphNode
import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.datatransform.SingleDataTransform
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.description.TransformDescription
import com.analyticspot.ml.framework.testutils.LowerCaseTransform
import com.fasterxml.jackson.annotation.JacksonInject
import com.fasterxml.jackson.annotation.JsonCreator
import org.assertj.core.api.Assertions.assertThat
import org.slf4j.LoggerFactory
import org.testng.annotations.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.util.Random
import java.util.concurrent.CompletableFuture
import java.util.concurrent.Executors

class MultiFileMixedFormatTest {
    companion object {
        private val log = LoggerFactory.getLogger(MultiFileMixedFormatTest::class.java)
    }

    // Here we create a full graph that we serialize. We ensure that there are nodes in the graph **after** our binary
    // data node so that we can ensure that the input/output streams for the nested zip are properly closed.
    @Test
    fun testSerializeGraphWithBinaryData() {
        val sillyOutput = "FAZZLE"
        val srcCol = ColumnId.create<String>("src")
        val sillyNode = SillyBinaryExample(sillyOutput)

        val dg = DataGraph.build {
            val src = setSource {
                columnIds += srcCol
            }

            val silly = addTransform(src, sillyNode)

            val lc = addTransform(silly, LowerCaseTransform(silly.transformDescription))

            result = lc
        }

        val serDeser = GraphSerDeser()
        serDeser.registerFormat(MultiFileMixedFormat())
        val output = ByteArrayOutputStream()

        log.info("Serializing the DataGraph")
        serDeser.serialize(dg, output)
        log.info("Serialization complete.")

        log.info("Deserializing the DataGraph")
        val dgDeser = serDeser.deserialize(ByteArrayInputStream(output.toByteArray()))
        log.info("Done deserializing the DataGraph")

        val sillyGraphNodeDeser = dgDeser.allNodes.firstOrNull {
            it is HasTransformGraphNode<*> && it.transform is SillyBinaryExample
        }
        assertThat(sillyGraphNodeDeser).isNotNull()
        sillyGraphNodeDeser as HasTransformGraphNode<*>
        val sillyNodeDeser = sillyGraphNodeDeser.transform as SillyBinaryExample
        assertThat(sillyNodeDeser.randomBytes).isEqualTo(sillyNode.randomBytes)

        val resultDs = dg.transform(dg.createSource("foo"), Executors.newSingleThreadExecutor()).get()
        assertThat(resultDs.column(sillyNode.outColumn)).containsExactly(sillyOutput.toLowerCase())
    }

    class SillyBinaryExample : SingleDataTransform, MultiFileMixedTransform {
        val outColumn = ColumnId.create<String>("foo")
        override val description: TransformDescription = TransformDescription(listOf(outColumn))
        val randomBytes: Array<Byte>
        val transformOutput: String

        @JsonCreator
        constructor(transformOutput: String,
                @JacksonInject(MultiFileMixedFormat.INJECTED_BINARY_DATA) binaryData: InputStream) {
            this.transformOutput = transformOutput
            randomBytes = Array<Byte>(NUM_BINARY_BYTES) { 0 }
            binaryData.read(randomBytes.toByteArray())
        }

        constructor(transformOutput: String) {
            this.transformOutput = transformOutput
            val r = Random()
            randomBytes = Array<Byte>(NUM_BINARY_BYTES) { 0 }
            r.nextBytes(randomBytes.toByteArray())
        }

        companion object {
            val NUM_BINARY_BYTES = 100
        }

        override fun transform(dataSet: DataSet): CompletableFuture<DataSet> {
            val resultList = mutableListOf<String>()
            0.until(dataSet.numRows).forEach { resultList.add(transformOutput) }
            return CompletableFuture.completedFuture(DataSet.create(outColumn, resultList))
        }

        override fun serializeBinaryData(output: OutputStream) {
            output.write(randomBytes.toByteArray())
        }

    }
}
