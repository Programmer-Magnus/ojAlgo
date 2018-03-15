package org.ojalgo.scalar;

import org.ojalgo.FunctionalityTest;
import org.ojalgo.TestUtils;
import org.ojalgo.constant.PrimitiveMath;

import java.math.BigDecimal;

public class RationalNumberTest extends FunctionalityTest {

    private final double myDiff = PrimitiveMath.MACHINE_EPSILON;

    public void testValueOf() {

        double test_values[] = {
                0.1,
                0.3,
                0.25,
                1e7,
                5e8,
                -25.22e-4,
                0.04919653065050689,
                1.2325077080153841
//                ,
//                4223372036854775807.0,
//                -4223372036854775808.0,
//                9223372036854775807.0,
//                -9223372036854775808.0
        };

        for (double d : test_values) {
            final RationalNumber direct = RationalNumber.valueOf(d);
            final RationalNumber viaBigDecimal = RationalNumber.valueOf(BigDecimal.valueOf(d));

            TestUtils.assertEquals(viaBigDecimal.doubleValue(), direct.doubleValue(), myDiff);
        }
    }

    public void testMultiplication() {
        RationalNumber a = RationalNumber.valueOf(0.04919653065050689);
        RationalNumber b = RationalNumber.valueOf(1.2325077080153841);

        TestUtils.assertEquals(a.multiply(b).doubleValue(), a.doubleValue() * b.doubleValue(), myDiff);
    }
}
