package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datagraph.AddConstantTransform
import com.analyticspot.ml.framework.datagraph.DataGraph
import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.datatransform.SupervisedLearningTransform
import com.analyticspot.ml.framework.description.ColumnId
import org.assertj.core.api.Assertions.assertThat
import org.testng.annotations.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class DelegatingFormatTest {
    @Test
    fun testWorks() {
        val dg = DataGraph.build {
            val src = dataSetSource()
            result = addTransform(src, src, Dtrans())
        }

        val colId = ColumnId.create<Int>("c1")
        val ds = DataSet.build {
            addColumn(colId, listOf(1, 2, 3, 4))
        }

        dg.trainTransform(ds, Executors.newSingleThreadExecutor()).get()
        val sds = GraphSerDeser()

        val out = ByteArrayOutputStream()
        sds.serialize(dg, out)

        val deser = sds.deserialize(ByteArrayInputStream(out.toByteArray()))

        val resultDs = deser.transform(ds, Executors.newSingleThreadExecutor()).get()

        assertThat(resultDs.column(colId)).containsExactlyElementsOf(
                ds.column(colId).mapToColumn { it!! + Dtrans.constantToAdd })
    }

    class Dtrans : SupervisedLearningTransform, DelegatingTransform {
        override lateinit var delegate: AddConstantTransform

        companion object {
            val constantToAdd = 18
        }

        override fun transform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
            return delegate.transform(dataSet, exec)
        }

        override fun trainTransform(dataSet: DataSet, targetDs: DataSet, exec: ExecutorService)
                : CompletableFuture<DataSet> {
            delegate = AddConstantTransform(constantToAdd)
            return delegate.transform(dataSet, exec)
        }

    }
}
