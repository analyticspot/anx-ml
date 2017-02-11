package com.analyticspot.ml.framework.feature

/**
 * A feature that can take on only two values, `true` or `false.
 */
class BooleanFeatureId(name: String, maybeMissing: Boolean) : FeatureId<Boolean>(name, Boolean::class, maybeMissing)
