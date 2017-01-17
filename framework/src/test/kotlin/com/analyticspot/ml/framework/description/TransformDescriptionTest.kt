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

package com.analyticspot.ml.framework.description

import com.analyticspot.ml.framework.serialization.JsonMapper
import org.assertj.core.api.Assertions.assertThat
import org.slf4j.LoggerFactory
import org.testng.annotations.Test

class TransformDescriptionTest {
    companion object {
        private val log = LoggerFactory.getLogger(TransformDescriptionTest::class.java)
    }

    @Test
    fun testCanSerializeAndDeserialize() {
        val td = TransformDescription(
                listOf(ColumnId.create<String>("c1"), ColumnId.create<Int>("c2")),
                listOf(ColumnIdGroup.create<String>("g1"), ColumnIdGroup.create<Double>("c2"))
        )

        val serialized = JsonMapper.mapper.writeValueAsString(td)
        log.debug("Serialized as: {}", serialized)
        val deserialized = JsonMapper.mapper.readValue(serialized, TransformDescription::class.java)

        assertThat(deserialized).isEqualTo(td)
    }
}
