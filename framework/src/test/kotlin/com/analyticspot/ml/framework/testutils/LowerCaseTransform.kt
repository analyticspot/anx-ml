/*
 * Copyright (C) 2017 Analytic Spot.
 * 
 * This file is part of the ANX ML library.
 * 
 * The ANX ML library is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser
 * General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 * 
 * Foobar is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License along with the ANX ML libarary.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

package com.analyticspot.ml.framework.testutils

import com.analyticspot.ml.framework.datatransform.SingleItemDataTransform
import com.analyticspot.ml.framework.description.TransformDescription

/**
 * Converts all the `String` type columns in a [DataSet] to lowercase.
 */
class LowerCaseTransform(srcTransDescription: TransformDescription)
    : SingleItemDataTransform<String, String>(srcTransDescription, String::class, String::class) {

    override fun transformItem(input: String): String {
        return input.toLowerCase()
    }
}
