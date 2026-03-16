package org.ojalgo.matrix.decomposition;

import org.junit.jupiter.api.Test;
import org.ojalgo.matrix.MatrixR128;
import org.ojalgo.matrix.store.RawStore;
import org.ojalgo.scalar.Quadruple;

import java.time.Duration;

import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class IssueSingularValueR128 {

    private void doTest(final RawStore matrix) {

        assertTimeoutPreemptively(Duration.ofSeconds(1), () -> {
            SingularValue<Double> svd = SingularValue.R064.make(matrix);
            boolean result = svd.decompose(matrix);
            assertTrue(result);
        }, "R064: Decomposition took too long, likely an infinite loop.");

        assertTimeoutPreemptively(Duration.ofSeconds(1), () -> {
            SingularValue<Quadruple> svd = SingularValue.R128.make(matrix);
            boolean result = svd.decompose(MatrixR128.FACTORY.copy(matrix));
            assertTrue(result);
        }, "R128: Decomposition took too long, likely an infinite loop.");
    }

    @Test
    public void testInfiniteLoopR128_0() {
        MatrixR128 matrix = MatrixR128.FACTORY.copy(RawStore.wrap(new double[][]{
                {1, 0, 0, 0},
                {0, 1, 1, 0},
                {0, 0, 1, 0},
                {0, 0, 0, 1}
        }));

        SingularValue<Quadruple> svd = SingularValue.R128.make(matrix);

        assertTimeoutPreemptively(Duration.ofSeconds(100000), () -> {
            boolean result = svd.decompose(matrix);
            assertTrue(result);
        }, "Decomposition took too long, likely an infinite loop.");
    }

    final double e = Double.MIN_VALUE;

    @Test
    public void testInfiniteLoopR128_1() {
        this.doTest(RawStore.wrap(new double[][] {
                { 1, e, e, e },
                { e, e, 1, e },
                { e, 1, e, e },
                { e, e, e, 1 }
        }));
    }

    @Test
    public void testInfiniteLoopR128_2() {
        this.doTest(RawStore.wrap(new double[][] {
                { 1, 0, 0, 0 },
                { 0, 0, 1, 0 },
                { 0, 1, 0, 0 },
                { 0, 0, 0, 1 }
        }));
    }

    @Test
    public void testInfiniteLoopR128_3() {
        this.doTest(RawStore.wrap(new double[][] {
                { 0, 0, 3, 0 },
                { 0, 1, 1, 1 },
                { 3, 1, 0, 0 },
                { 0, 1, 0, 0 }
        }));
    }

    @Test
    public void testInfiniteLoopR128_4() {
        this.doTest(RawStore.wrap(new double[][] {
                { 0, 0, 3, 0 },
                { 0, 12433.182577348916, 1, 0.3545802800218381 },
                { 3, 1, 0, -2.8191084267847077E-24 },
                { 0, 0.15640108192643198, 0, 3.2230232111154273E-8 }
        }));
    }
}
