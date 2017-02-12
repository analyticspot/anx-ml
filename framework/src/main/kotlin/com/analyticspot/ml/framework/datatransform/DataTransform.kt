/*
 * Copyright (C) 2017 Analytic Spot.
 * 
 * This file is part of the ANX ML library.
 * 
 * The ANX ML library is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser
 * General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 * 
 * The ANX ML library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
 * for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License along with the ANX ML libarary.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.serialization.Format
import com.analyticspot.ml.framework.serialization.StandardJsonFormat
import com.fasterxml.jackson.annotation.JsonIgnore

/**
 * Base interface for all tranformations.
 */
interface DataTransform {
    /**
     * The format to which this node serializes. By default this is [StandardJsonFormat].
     */
    val formatClass: Class<out Format<*>>
        @JsonIgnore
        get() = StandardJsonFormat::class.java
}

