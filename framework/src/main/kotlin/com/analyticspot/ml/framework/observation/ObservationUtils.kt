package com.analyticspot.ml.framework.observation

import com.analyticspot.ml.framework.description.ValueToken

/**
 * Various utility functions for working with [Observation]s.
 */

/**
 * Returns true if and only if `a.value(x) == b.value(x)` for all `x` in `toCheck`.
 */
fun equalValues(toCheck: List<ValueToken<*>>, a: Observation, b: Observation): Boolean {
    for (tok in toCheck) {
        if (a.value(tok) != b.value(tok)) {
            return false
        }
    }
    return true
}
