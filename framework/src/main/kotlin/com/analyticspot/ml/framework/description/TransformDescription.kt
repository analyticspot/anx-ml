package com.analyticspot.ml.framework.description

/**
 * Describes the outputs produced by an execution of the node. This allows each [GraphNode] to tell subscribing nodes
 * what columns are available.
 *
 * @param columns These are the columns **that can be known before training**. Some transforms can not know how many
 *     columns they will produce until they have been trained (e.g. a bag-of-words transform can't know how many words
 *     are in the vocabulary until it's been trained).
 * @param columnGroups These allow transforms which can't know how many columns they will produce to provide a way to
 *     reference the full set that will be produced. For example, a bag-of-words transform can't know the vocabulary
 *     size until it's trained, but it knows it will produce one column for each unique word in the input and can thus
 *     provide a [ColumnIdGroup] to allow access to all the generated columns.
 */
class TransformDescription(val columns: List<ColumnId<*>>,
        val columnGroups: List<ColumnIdGroup<*>> = listOf()) {
}
