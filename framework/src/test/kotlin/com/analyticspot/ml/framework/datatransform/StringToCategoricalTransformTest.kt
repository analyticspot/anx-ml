package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import org.assertj.core.api.Assertions.assertThat
import org.testng.annotations.Test
import java.util.concurrent.Executors

class StringToCategoricalTransformTest {
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
}
