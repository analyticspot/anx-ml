package com.analyticspot.ml.briges.deeplearn

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import org.assertj.core.api.Assertions.assertThat
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator
import org.nd4j.linalg.dataset.api.MultiDataSet
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.slf4j.LoggerFactory
import org.testng.annotations.Test
import java.util.Random

class RandomizingMultiDataSetIteratorTest {
    companion object {
        private val log = LoggerFactory.getLogger(RandomizingMultiDataSetIteratorTest::class.java)

        val SAMPLE_DATA_RESOURCE = "/DataSetIteratorData.csv"
    }
    // DeepLearning4j supplies several MultiDataSetIterator implementations (that all read from files and don't work
    // with in-memory data and don't work with our DataSet instances). We want to test that we produce the same exact
    // results as they do on the same data. So we have a CSV file we can use with DL4j and we then manually read that
    // CSV file into a DataSet and build our own MutliDataSetIterator and ensure they all produce the same stuff. The
    // input file contains 6 columns. The first two are double and will be one DataSet in our MulitDataSet, the 2nd 2
    // are a double and an integer. The 5th and 6th are integers that represent the target and will be one-hot encoded.
    @Test
    fun testProducesSameResultAsDl4j() {
        val totalColumns = 6
        val numRows = 20
        // Intentionally setting a batch size that doesn't evenly divide into the number of rows so we can test the
        // final, partial batch.
        val batchSize = 12

        // allData is a list of columns, each of type Double (we'll convert the targets to int later)
        val allData: MutableList<MutableList<Double>> = mutableListOf()
        for (i in 0 until totalColumns) {
            allData.add(mutableListOf())
        }

        this.javaClass.getResourceAsStream(SAMPLE_DATA_RESOURCE).bufferedReader().useLines { lineSequence ->
            lineSequence.forEach {
                val parts = it.split(",")
                check(parts.size == 6)
                parts.forEachIndexed { colIdx, part ->
                    allData[colIdx].add(part.toDouble())
                }
            }
        }

        val ds1 = DataSet.build {
            addColumn(ColumnId.create<Double>("c11"), allData[0])
            addColumn(ColumnId.create<Double>("c12"), allData[1])
        }

        val ds2 = DataSet.build {
            addColumn(ColumnId.create<Double>("c21"), allData[2])
            addColumn(ColumnId.create<Double>("c22"), allData[3])
        }

        val dsTargets = DataSet.build {
            addColumn(ColumnId.create<Int>("t1"), allData[4].map { it.toInt() })
            addColumn(ColumnId.create<Int>("t2"), allData[5].map { it.toInt() })
        }

        check(ds1.numRows == numRows)

        val ourIter = RandomizingMultiDataSetIterator(batchSize, listOf(ds1, ds2), dsTargets)

        val dl4jReader = CSVRecordReader(0, ",")
        dl4jReader.initialize(FileSplit(ClassPathResource(SAMPLE_DATA_RESOURCE).file))
        val dl4jIter = RecordReaderMultiDataSetIterator.Builder(batchSize)
                .addReader("fazzle", dl4jReader)
                .addInput("fazzle", 0, 1)
                .addInput("fazzle", 2, 3)
                .addOutputOneHot("fazzle", 4, 4)
                .addOutputOneHot("fazzle", 5, 4)
                .build()

        // Both iterators randomize the rows so we can't directly compare the data returned by the iterators.
        // However, they should return the same **set** of rows so we'll save all the rows we get in a map keyed by the
        // first value in the first data set (which is unique). We do this for our iterator and the dl4j one and we then
        // check that the maps are equal.

        val ourRows = mutableMapOf<Double, List<Double>>()
        val dl4jRows = mutableMapOf<Double, List<Double>>()

        while (ourIter.hasNext()) {
            log.debug("Checking a batch")
            assertThat(dl4jIter.hasNext())
            val ourMds = ourIter.next()
            addRowsToMap(ourMds, ourRows)

            val dl4jMds = dl4jIter.next()
            addRowsToMap(dl4jMds, dl4jRows)

            val ourFeatures = ourMds.features
            val dl4jFeatures = dl4jMds.features
            assertThat(ourFeatures).hasSameSizeAs(dl4jFeatures)
            for (fidx in ourFeatures.indices) {
                assertThat(ourFeatures[fidx].shape()).isEqualTo(dl4jFeatures[fidx].shape())
            }

            val ourTargets = ourMds.labels
            val dl4jTargets = dl4jMds.labels
            assertThat(ourTargets).hasSameSizeAs(dl4jTargets)
            for (tidx in ourTargets.indices) {
                assertThat(ourTargets[tidx].shape()).isEqualTo(dl4jTargets[tidx].shape())
            }
        }
        assertThat(ourIter.hasNext()).isFalse()
        assertThat(dl4jIter.hasNext()).isFalse()

        assertThat(ourRows).isEqualTo(dl4jRows)

        // And make sure they both reset the same way
        dl4jIter.reset()
        ourIter.reset()
        assertThat(dl4jIter.hasNext()).isTrue()
        assertThat(ourIter.hasNext()).isTrue()
    }

    @Test
    fun testIteratorRandomizesOnReset() {
        val rng = Random(12345)
        val numRows = 100
        val maxTargetVal = 7

        val ds1 = DataSet.build {
            addColumn(ColumnId.create<Double>("c"), (0 until numRows).map { rng.nextDouble() })
        }

        val targs = DataSet.build {
            addColumn(ColumnId.create<Int>("t"), (0 until numRows).map { rng.nextInt(maxTargetVal) })
        }

        val iter = RandomizingMultiDataSetIterator(10, listOf(ds1), targs, rng)

        val beforeReset = iterToList(iter)

        iter.reset()

        val afterReset = iterToList(iter)

        // Now both should hold exactly the same contents but in different orders
        assertThat(beforeReset).hasSameElementsAs(afterReset)
        val sameOrder = beforeReset.zip(afterReset).all { it.first == it.second }
        assertThat(sameOrder).isFalse()
    }

    private fun addRowsToMap(data: MultiDataSet, map: MutableMap<Double, List<Double>>) {
        val dataList = dataToList(data)
        dataList.forEach {
            val key: Double = it[0]
            map.put(key, it)
        }
    }

    // Concatenate all the feature data sets and all the target data sets to list of lists.
    private fun dataToList(data: MultiDataSet): List<List<Double>> {
        val result = mutableListOf<List<Double>>()
        for (i in 0 until data.features[0].rows()) {
            val completeRow = mutableListOf<Double>()
            for (feat in data.features) {
                for (col in 0 until feat.columns()) {
                    completeRow.add(feat.getDouble(i, col))
                }
            }

            for (targ in data.labels) {
                for (col in 0 until targ.columns()) {
                    completeRow.add(targ.getDouble(i, col))
                }
            }
            result.add(completeRow)
        }
        return result
    }

    // Like dataToList but for all the data sets returned by the iterator
    private fun iterToList(iter: MultiDataSetIterator): List<List<Double>> {
        val result = mutableListOf<List<Double>>()
        for (mds in iter) {
            val dataList = dataToList(mds)
            result.addAll(dataList)
        }
        return result
    }
}
