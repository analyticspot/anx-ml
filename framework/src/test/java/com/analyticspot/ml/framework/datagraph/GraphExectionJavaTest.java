package com.analyticspot.ml.framework.datagraph;

import static org.assertj.core.api.Assertions.assertThat;

import com.analyticspot.ml.framework.dataset.DataSet;
import com.analyticspot.ml.framework.dataset.DataSetUtilsKt;
import com.analyticspot.ml.framework.description.ValueId;
import com.analyticspot.ml.framework.description.ValueToken;
import com.analyticspot.ml.framework.testutils.LowerCaseTransform;
import com.analyticspot.ml.framework.testutils.TrueIfSeenTransform;
import org.assertj.core.util.Lists;
import org.testng.annotations.Test;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

/**
 * This is testing Java interoperability. Most of the real tests are in Kotlin.
 */
public class GraphExectionJavaTest {
  @Test
  public void testCanCreateAndExecuteAGraph() throws Exception {
    DataGraph.GraphBuilder graphBuilder = DataGraph.builder();

    ValueId<String> wordsId = new ValueId<>("words", String.class);
    ValueId<Boolean> targetId = new ValueId<>("target", Boolean.class);
    ValueId<Boolean> predictionId = new ValueId<>("prediction", Boolean.class);

    GraphNode source = graphBuilder.source()
        .withValue(wordsId)
        .withTrainOnlyValue(targetId)
        .build();

    GraphNode toLower = graphBuilder.addTransform(source,
        new LowerCaseTransform(source.token(wordsId), wordsId));

    GraphNode trueIfSeen = graphBuilder.addTransform(toLower, source,
        new TrueIfSeenTransform(toLower.token(wordsId), source.token(targetId), predictionId));

    graphBuilder.setResult(trueIfSeen);

    DataGraph graph = graphBuilder.build();

    DataSet trainingData = DataSetUtilsKt.createDataSet(
        graph.buildSourceObservation("foO", true),
        graph.buildSourceObservation("bar", false),
        graph.buildSourceObservation("BAZ", true)
    );

    CompletableFuture<DataSet> trainResult = graph.trainTransform(
        trainingData, Executors.newFixedThreadPool(4));

    ValueToken<Boolean> predictionToken = graph.getResult().token(predictionId);

    List<Boolean> trainPredictions = DataSetUtilsKt.toStream(trainResult.get())
        .map(obs -> obs.value(predictionToken)).collect(Collectors.toList());

    assertThat(trainPredictions).isEqualTo(Lists.newArrayList(true, false, true));

    DataSet testData = DataSetUtilsKt.createDataSet(
        graph.buildSourceObservation("FOO"),
        graph.buildSourceObservation("bip"),
        graph.buildSourceObservation("baz"),
        graph.buildSourceObservation("blah")
    );

    DataSet testResult = graph.transform(testData, Executors.newFixedThreadPool(4)).get();
    List<Boolean> testPredictions = DataSetUtilsKt.toStream(testResult)
        .map(obs -> obs.value(predictionToken)).collect(Collectors.toList());
    assertThat(testPredictions).isEqualTo(Lists.newArrayList(true, false, true, false));
  }
}
