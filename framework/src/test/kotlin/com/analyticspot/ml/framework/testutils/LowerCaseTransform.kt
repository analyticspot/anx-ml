package com.analyticspot.ml.framework.testutils

import com.analyticspot.ml.framework.datatransform.PerItemDataTransform
import com.analyticspot.ml.framework.description.AggregateValueIdGroup
import com.analyticspot.ml.framework.description.ValueId
import com.analyticspot.ml.framework.description.ValueIdGroup
import com.analyticspot.ml.framework.description.ValueToken
import com.analyticspot.ml.framework.description.ValueTokenGroup
import com.analyticspot.ml.framework.description.groupFromToken

/**
 * Takes a String input and converts it to lowercase.
 */
class LowerCaseTransform(srcGroup: ValueTokenGroup<String>,
        resultId: ValueIdGroup<String>)
    : PerItemDataTransform<String, String>(srcGroup, resultId, String::class.java) {

    /**
     * Convenience constructor for a transform of a single input.
     */
    constructor(srcTok: ValueToken<String>, resultId: ValueId<String>) : this(
            groupFromToken(srcTok.name, srcTok),
            AggregateValueIdGroup.build<String>(resultId.name) { valueIds += resultId })


    override fun transform(input: String): String {
        return input.toLowerCase()
    }

}
