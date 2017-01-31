package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datagraph.GraphNode
import com.analyticspot.ml.framework.datatransform.DataTransform
import com.analyticspot.ml.framework.serialization.MultiFileMixedFormat.Companion.INJECTED_BINARY_DATA
import com.fasterxml.jackson.databind.InjectableValues
import org.slf4j.LoggerFactory
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream

/**
 * A format for classes that have a mix of JSON data and some kind of binary data. For example, when we wrap
 * DeepLearning4j algorithms we need to serialize some data in our wrapper, but we also have to serialize the underlying
 * DeepLearning4j algorithm and those serialize to a binary format.
 *
 * For serialization classes must implement the [MultiFileMixedTransform] interface. For deserialization an
 * `InputStream` will be available via `@JacksonInject` using the key [INJECTED_BINARY_DATA] so you can annotate a
 * property or function in the class to be deserialized with `@JacksonInject(INJECTED_BINARY_DATA)`.
 */
class MultiFileMixedFormat() : Format<MultiFileMixedFormat.MetaData> {
    override val metaDataClass: Class<MetaData> = MetaData::class.java

    companion object {
        private val log = LoggerFactory.getLogger(MultiFileMixedFormat::class.java)

        private val JSON_PART_NAME = "jsonPart"
        private val BINARY_PART_NAME = "binaryPart"

        /**
         * During deserialization the `InputStream` holding the binary data will be made available via Jackson's
         * `@JacksonInject` so if you annotate a method or property class to be deserialized with
         * `@JacksonInject(INJECTED_BINARY_DATA)` you will get the input stream provided.
         */
        const val INJECTED_BINARY_DATA: String = "binaryBlob"
    }

    override fun getMetaData(transform: DataTransform): MetaData {
        return MetaData(transform.javaClass)
    }

    override fun serialize(transform: DataTransform, output: OutputStream) {
        if (transform is MultiFileMixedTransform) {
            // Create a nested zip file with 2 parts. Part 1 is the JSON serialization of the data and part2 is the binary
            // blob.
            val zipOut = ZipOutputStream(output)

            zipOut.putNextEntry(ZipEntry(BINARY_PART_NAME))
            transform.serializeBinaryData(zipOut)
            zipOut.closeEntry()

            zipOut.putNextEntry(ZipEntry(JSON_PART_NAME))
            JsonMapper.mapper.writeValue(zipOut, transform)
            zipOut.closeEntry()

            zipOut.finish()
        } else {
            throw IllegalArgumentException(
                    "$transform is not a MultipartMixedTransform and so can not use MultiFileMixedFormat")
        }
    }

    override fun deserialize(metaData: MetaData, sources: List<GraphNode>, input: InputStream): DataTransform {
        val zipIn = ZipInputStream(input)

        // First read the binary data into a Byte array
        val entry = zipIn.nextEntry
        check(entry.name == BINARY_PART_NAME)
        val binaryData = readInputStreamToOutputStream(zipIn)
        zipIn.closeEntry()

        // Now deserialize the JSON stuff but with the byte array available via injection
        val jsonEntry = zipIn.nextEntry
        check(jsonEntry.name == JSON_PART_NAME)
        val toInject = InjectableValues.Std()
                .addValue(INJECTED_BINARY_DATA, ByteArrayInputStream(binaryData.toByteArray()))
        val result = JsonMapper.mapper.setInjectableValues(toInject).readValue(zipIn, metaData.transformClass)
        zipIn.closeEntry()
        return result
    }

    private fun readInputStreamToOutputStream(input: InputStream): ByteArrayOutputStream {
        val bufferSize = 1024
        val buffer = ByteArrayOutputStream()

        val data = ByteArray(bufferSize)

        while (true) {
            var nRead: Int = input.read(data)
            if (nRead == -1) {
                break
            }
            check(nRead > 0)
            buffer.write(data, 0, nRead)
        }
        buffer.flush()
        return buffer
    }


    class MetaData(val transformClass: Class<out DataTransform>) : FormatMetaData {
        override val formatClass = MultiFileMixedFormat::class.java

    }
}
