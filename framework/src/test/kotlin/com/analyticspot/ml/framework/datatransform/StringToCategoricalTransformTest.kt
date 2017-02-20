package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.metadata.CategoricalFeatureMetaData
import com.analyticspot.ml.framework.serialization.GraphSerDeser
import com.analyticspot.ml.framework.serialization.JsonMapper
import org.assertj.core.api.Assertions.assertThat
import org.slf4j.LoggerFactory
import org.testng.annotations.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

class StringToCategoricalTransformTest {
    companion object {
        private val log = LoggerFactory.getLogger(StringToCategoricalTransformTest::class.java)
    }

    @Test
    fun testWorks() {
        val cid1 = ColumnId.create<String>("c1")
        val cid2 = ColumnId.create<String>("c2")
        val cid3 = ColumnId.create<String>("c3")

        val dsTrain = DataSet.build {
            addColumn(cid1, listOf("a", "b", "c", "d"))
            addColumn(cid2, listOf("foo", "baz", "baz", "foo"))
            addColumn(cid3, listOf("will", "not", "be", "transformed"))
        }

        // Transform the 1st 2 columns, ignore the 3rd
        val trans = StringToCategoricalTransform(cid1, cid2)
        val exec = Executors.newSingleThreadExecutor()

        val dsTrainResult = trans.trainTransform(dsTrain, exec).get()

        // The output should not contain column 3
        assertThat(dsTrainResult.numColumns).isEqualTo(2)
        assertThat(dsTrainResult.column(cid1)).containsExactlyElementsOf(dsTrain.column(cid1))
        assertThat(dsTrainResult.column(cid2)).containsExactlyElementsOf(dsTrain.column(cid2))

        val m1 = dsTrainResult.metaData[cid1.name]
        assertThat(m1).isInstanceOf(CategoricalFeatureMetaData::class.java)
        val m1AsCat = m1 as CategoricalFeatureMetaData
        assertThat(m1AsCat.maybeMissing).isTrue()

        val m2 = dsTrainResult.metaData[cid2.name]
        assertThat(m2).isInstanceOf(CategoricalFeatureMetaData::class.java)
        val m2AsCat = m2 as CategoricalFeatureMetaData
        assertThat(m2AsCat.maybeMissing).isTrue()

        val dsTransform = DataSet.build {
            addColumn(cid1, listOf("b", "c", "f", "a"))
            addColumn(cid2, listOf("baz", "bar", "foo", "bar"))
            addColumn(cid3, listOf("will", "not", "be", "transformed"))
        }

        val dsTransformResult = trans.transform(dsTransform, exec).get()
        assertThat(dsTransformResult.numColumns).isEqualTo(2)
        assertThat(dsTransformResult.column(cid1)).containsExactly("b", "c", null, "a")
        assertThat(dsTransformResult.column(cid2)).containsExactly("baz", null, "foo", null)
    }

    @Test
    fun testCanSerDeser() {
        val cid1 = ColumnId.create<String>("c1")
        val cid2 = ColumnId.create<String>("c2")
        val cid3 = ColumnId.create<String>("c3")

        val dsTrain = DataSet.build {
            addColumn(cid1, listOf("a", "b", "c", "d"))
            addColumn(cid2, listOf("foo", "baz", "baz", "foo"))
            addColumn(cid3, listOf("will", "not", "be", "transformed"))
        }

        // Transform the 1st 2 columns, ignore the 3rd
        val trans = StringToCategoricalTransform(cid1, cid2)
        val exec = Executors.newSingleThreadExecutor()

        trans.trainTransform(dsTrain, exec).get()

        val serDeser = GraphSerDeser()
        val output = ByteArrayOutputStream()
        serDeser.serializeTransform(trans, output)

        log.info("Model serialized as: {}", output.toString(Charsets.UTF_8.displayName()))

        val deserTrans = JsonMapper.mapper.readValue(
                ByteArrayInputStream(output.toByteArray()), StringToCategoricalTransform::class.java)

        val dsTransform = DataSet.build {
            addColumn(cid1, listOf("b", "c", "f", "a"))
            addColumn(cid2, listOf("baz", "bar", "foo", "bar"))
            addColumn(cid3, listOf("will", "not", "be", "transformed"))
        }

        // The deserialized transform should remember the metadata, etc. and act just like the regular non-serialized
        // transform did in the test above.
        val dsTransformResult = deserTrans.transform(dsTransform, exec).get()
        assertThat(dsTransformResult.numColumns).isEqualTo(2)
        assertThat(dsTransformResult.column(cid1)).containsExactly("b", "c", null, "a")
        assertThat(dsTransformResult.column(cid2)).containsExactly("baz", null, "foo", null)

        val m1 = dsTransformResult.metaData[cid1.name]
        assertThat(m1).isInstanceOf(CategoricalFeatureMetaData::class.java)
        val m1AsCat = m1 as CategoricalFeatureMetaData
        assertThat(m1AsCat.maybeMissing).isTrue()

        val m2 = dsTransformResult.metaData[cid2.name]
        assertThat(m2).isInstanceOf(CategoricalFeatureMetaData::class.java)
        val m2AsCat = m2 as CategoricalFeatureMetaData
        assertThat(m2AsCat.maybeMissing).isTrue()
    }
}
