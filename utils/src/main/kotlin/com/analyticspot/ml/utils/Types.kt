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

package com.analyticspot.ml.utils

import kotlin.reflect.KClass

/**
 * Handy functions for working with Java and Kotlin type information.
 */

/**
 * Figuring out if two classes are compatible in Kotlin is a bit more complex than in Java because of classes like
 * `Int` and `Long` which are, in some situations, boxed and are, in others, not. But
 * `Integer.class.isAssignableFrom(Int::class.java)` will return false. So this checks if the equivalent Java types
 * are compatible and returns true if and only if you actually can assign an object of class `other` to the
 * receiver.
 */
infix fun KClass<*>.isAssignableFrom(other: KClass<*>): Boolean {
    return this.javaObjectType.isAssignableFrom(other.javaObjectType)
}

/**
 * The same as the other overload but works with Java class types.
 */
infix fun Class<*>.isAssignableFrom(other: Class<*>): Boolean {
    return this.kotlin isAssignableFrom other.kotlin
}

/**
 * The same as the other overload but works with a mix of Java and Kotlin class types.
 */
infix fun KClass<*>.isAssignableFrom(other: Class<*>): Boolean {
    return this isAssignableFrom other.kotlin
}

/**
 * The same as the other overload but works with a mix of Java and Kotlin class types.
 */
infix fun Class<*>.isAssignableFrom(other: KClass<*>): Boolean {
    return this.kotlin isAssignableFrom other
}
