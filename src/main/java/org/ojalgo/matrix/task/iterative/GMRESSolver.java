/*
 * Copyright 1997-2025 Optimatika
 */
package org.ojalgo.matrix.task.iterative;

import static org.ojalgo.function.constant.PrimitiveMath.*;

import java.util.List;

import org.ojalgo.equation.Equation;
import org.ojalgo.matrix.store.PhysicalStore;
import org.ojalgo.matrix.store.R064Store;
import org.ojalgo.structure.Mutate1D;
import org.ojalgo.type.context.NumberContext;

/**
 * Generalized Minimal RESidual (GMRES) solver for general nonsymmetric square systems.
 * <p>
 * This is a Java port of SciPy's gmres() reference implementation. It is almost a straight line-by-line
 * translation from SciPy. Implemented here in an unpreconditioned form (M=I) and using ojAlgo's dense
 * MatrixStore operations for matrix–vector products with A.
 * <p>
 * When to use:
 * <ul>
 * <li>Nonsymmetric or indefinite systems where ConjugateGradient is not applicable.</li>
 * <li>When you need a robust Krylov method that minimizes the residual over the Krylov subspace.</li>
 * <li>Prefer over Jacobi/Gauss–Seidel for difficult nonsymmetric problems requiring better convergence.</li>
 * <li>GMRES typically converges faster than BiCGSTAB or QMR but requires more memory.</li>
 * </ul>
 * Characteristics:
 * <ul>
 * <li>Requires only A·x products (no transpose needed, unlike QMR).
 * <li>No preconditioning (M=I).
 * <li>This version does not work with complex numbers.
 * <li>Uses restart mechanism: after a fixed number of iterations (default 20), restart with current solution.
 * <li>The stopping criterion follows ojAlgo style: terminate when NumberContext.isSmall(||b||, ||r||).
 * <li>Designed for square systems; Throw IllegalArgumentException for non-square inputs.
 * </ul>
 * References:
 * <ul>
 * <li>SciPy 1.16.1 implementation: scipy.sparse.linalg._isolve.iterative.gmres
 * <li>https://www.netlib.org/templates/templates.pdf, Figure 2.9.
 * (https://github.com/scipy/scipy/blob/0cf8e9541b1a2457992bf4ec2c0c669da373e497/scipy/sparse/linalg/_isolve/iterative.py)
 * <li>Saad, Youcef, and Martin H. Schultz. "GMRES: A generalized minimal residual algorithm for solving
 * nonsymmetric linear systems." SIAM Journal on scientific and statistical computing 7.3 (1986): 856-869.
 * </ul>
 */
public final class GMRESSolver extends IterativeSolverTask {

    private static void axpy(final double alpha, final R064Store x, final Mutate1D.Modifiable<?> y) {
        x.axpy(alpha, y);
    }

    private static double norm2(final R064Store a) {
        double n = ZERO;
        for (int i = 0; i < a.getRowDim(); i++) {
            n = HYPOT.invoke(n, a.doubleValue(i));
        }
        return n;
    }

    private static void scaleCopy(final R064Store src, final double alpha, final R064Store dst) {
        for (int i = 0; i < src.getRowDim(); i++) {
            dst.set(i, alpha * src.doubleValue(i));
        }
    }

    private static void scaleInPlace(final R064Store x, final double alpha) {
        x.modifyAll(MULTIPLY.by(alpha));
    }

    /** GMRES restart parameter (number of inner iterations before restart). */
    private int restart = 20;

    /** Krylov basis vectors v[0..restart]. */
    private R064Store[] v;

    /** Hessenberg matrix h[restart x restart+1]. */
    private R064Store h;

    /** Givens rotation parameters [restart x 2]: [c, s]. */
    private R064Store givens;

    /** Right-hand side of least squares problem. */
    private R064Store rhs;

    /** Temporary work vectors. */
    private R064Store w;
    private R064Store r;

    /** Temporary solution vector for back substitution. */
    private R064Store y;

    public GMRESSolver() {
        super();
    }

    /**
     * Set the restart parameter for GMRES. After this many iterations, the algorithm restarts with the
     * current solution. Default is 20.
     *
     * @param restart
     *            number of iterations before restart
     * @return this solver (for chaining)
     */
    public GMRESSolver restart(final int restart) {
        this.restart = Math.max(1, restart);
        return this;
    }

    @Override
    public double resolve(final List<Equation> equations, final PhysicalStore<Double> x) {

        if (this.isDebugPrinterSet()) {
            this.debug(0, NaN, x);
        }

        int m = equations.size();
        int n = x.size();

        // Adjust restart to not exceed problem size
        int restartActual = Math.min(restart, n);

        int nbIterations = 0;
        int iterationsLimit = this.getIterationsLimit();

        NumberContext accuracy = this.getAccuracyContext();

        // Allocate Krylov basis vectors v[0..restart]
        if (v == null || v.length < restartActual + 1) {
            v = new R064Store[restartActual + 1];
            for (int i = 0; i <= restartActual; i++) {
                v[i] = R064Store.FACTORY.make(n, 1);
            }
        } else {
            // Ensure existing vectors are correct size
            for (int i = 0; i <= restartActual; i++) {
                if (v[i] == null || v[i].getRowDim() != n) {
                    v[i] = R064Store.FACTORY.make(n, 1);
                }
            }
        }

        // Allocate Hessenberg matrix h[restart x restart+1]
        if (h == null || h.getRowDim() < restartActual || h.getColDim() < restartActual + 1) {
            h = R064Store.FACTORY.make(restartActual, restartActual + 1);
        }

        // Allocate Givens rotation parameters [restart x 2]: [c, s]
        if (givens == null || givens.getRowDim() < restartActual || givens.getColDim() < 2) {
            givens = R064Store.FACTORY.make(restartActual, 2);
        }

        // Allocate RHS vector for least squares problem
        rhs = IterativeSolverTask.worker(rhs, restartActual + 1);

        // Allocate temporary solution vector for back substitution
        y = IterativeSolverTask.worker(y, restartActual + 1);

        // Allocate work vectors
        w = IterativeSolverTask.worker(w, n);
        r = IterativeSolverTask.worker(r, n);

        // Compute initial residual r = b - A*x and norms
        double normRHS = ZERO;
        double normErr = ZERO;
        for (int i = 0; i < m; i++) {
            Equation row = equations.get(i);
            double bi = row.getRHS();
            normRHS = HYPOT.invoke(normRHS, bi);
            double ri = bi - row.dot(x);
            r.set(row.index, ri);
            normErr = HYPOT.invoke(normErr, ri);
        }

        if (this.isDebugPrinterSet()) {
            this.debug(0, NaN, x);
        }

        if (normErr == ZERO) {
            return ZERO;
        }

        // Machine epsilon for breakdown detection
        double eps = Math.ulp(ONE);

        // Outer loop: restart cycles
        while (nbIterations < iterationsLimit && !accuracy.isSmall(normRHS, normErr)) {

            // Convergence check on residual relative to RHS
            if (accuracy.isSmall(normRHS, normErr)) {
                break; // convergence
            }

            // Initialize first Krylov vector: v[0] = r / ||r||
            // In unpreconditioned case: v[0] = psolve(r) / ||psolve(r)|| = r / ||r||
            double beta = GMRESSolver.norm2(r);
            if (beta == ZERO) {
                break; // Already converged
            }
            GMRESSolver.scaleCopy(r, ONE / beta, v[0]);

            // Initialize RHS of least squares problem
            for (int i = 0; i <= restartActual; i++) {
                rhs.set(i, ZERO);
            }
            rhs.set(0, beta);

            // Clear Hessenberg matrix
            h.fillAll(ZERO);

            boolean breakdown = false;
            int col;

            // Inner loop: build Krylov subspace
            for (col = 0; col < restartActual; col++) {

                // Compute w = A * v[col] (unpreconditioned: w = psolve(A * v[col]) = A * v[col])
                w.fillAll(ZERO);
                for (int i = 0; i < m; i++) {
                    Equation row = equations.get(i);
                    double av = row.dot(v[col]);
                    w.set(row.index, w.doubleValue(row.index) + av);
                }

                // Modified Gram-Schmidt orthogonalization
                double h0 = GMRESSolver.norm2(w);
                for (int k = 0; k <= col; k++) {
                    // h[col, k] = <v[k], w>
                    double tmp = ZERO;
                    for (int i = 0; i < n; i++) {
                        tmp += v[k].doubleValue(i) * w.doubleValue(i);
                    }
                    h.set(col, k, tmp);
                    // w = w - h[col, k] * v[k]
                    GMRESSolver.axpy(-tmp, v[k], w);
                }

                // h[col, col+1] = ||w||
                double h1 = GMRESSolver.norm2(w);
                h.set(col, col + 1, h1);

                // Check for breakdown: exact solution found
                if (h1 <= eps * h0) {
                    h.set(col, col + 1, ZERO);
                    breakdown = true;
                    // v[col+1] remains unused
                } else {
                    // v[col+1] = w / h[col, col+1]
                    GMRESSolver.scaleCopy(w, ONE / h1, v[col + 1]);
                }

                // Apply previous Givens rotations to current h column
                for (int k = 0; k < col; k++) {
                    double c = givens.doubleValue(k, 0);
                    double s = givens.doubleValue(k, 1);
                    double n0 = h.doubleValue(col, k);
                    double n1 = h.doubleValue(col, k + 1);
                    h.set(col, k, c * n0 + s * n1);
                    h.set(col, k + 1, -s * n0 + c * n1);
                }

                // Compute and apply current Givens rotation
                // Rotate h[col, col] and h[col, col+1] to eliminate h[col, col+1]
                double hcc = h.doubleValue(col, col);
                double hcc1 = h.doubleValue(col, col + 1);
                double c, s, mag;
                if (hcc1 == ZERO) {
                    c = ONE;
                    s = ZERO;
                    mag = hcc;
                } else if (Math.abs(hcc1) > Math.abs(hcc)) {
                    double t = hcc / hcc1;
                    s = ONE / Math.sqrt(ONE + t * t);
                    c = s * t;
                    mag = hcc1 / s;
                } else {
                    double t = hcc1 / hcc;
                    c = ONE / Math.sqrt(ONE + t * t);
                    s = c * t;
                    mag = hcc / c;
                }
                givens.set(col, 0, c);
                givens.set(col, 1, s);
                h.set(col, col, mag);
                h.set(col, col + 1, ZERO);

                // Apply rotation to RHS
                double rhs_col = rhs.doubleValue(col);
                double rhs_col1 = rhs.doubleValue(col + 1);
                rhs.set(col, c * rhs_col + s * rhs_col1);
                rhs.set(col + 1, -s * rhs_col + c * rhs_col1);

                // Preconditioned residual norm (for inner convergence check)
                double presid = Math.abs(rhs.doubleValue(col + 1));

                nbIterations++;

                if (this.isDebugPrinterSet()) {
                    // Compute true residual for debug output
                    double tmpNormErr = ZERO;
                    for (int i = 0; i < m; i++) {
                        Equation row = equations.get(i);
                        double ri = row.getRHS() - row.dot(x);
                        tmpNormErr = HYPOT.invoke(tmpNormErr, ri);
                    }
                    this.debug(nbIterations, tmpNormErr / normRHS, x);
                }

                // Check for convergence or breakdown
                if (breakdown || nbIterations >= iterationsLimit) {
                    break;
                }
            }

            // Determine actual Krylov subspace dimension
            // If loop completed without break, col = restartActual, but last iteration was restartActual-1
            // If loop broke early, col is the iteration where we broke
            int krylovDim = (col < restartActual) ? col : restartActual - 1;

            // Solve upper triangular system H * y = rhs
            // H is (krylovDim+1) x (krylovDim+1) upper triangular stored in h[0..krylovDim][0..krylovDim]
            // Copy rhs to y for back substitution
            for (int i = 0; i <= krylovDim; i++) {
                y.set(i, rhs.doubleValue(i));
            }

            // Back substitution
            for (int k = krylovDim; k >= 0; k--) {
                double hkk = h.doubleValue(k, k);
                if (hkk != ZERO) {
                    double yk = y.doubleValue(k) / hkk;
                    y.set(k, yk);
                    for (int i = 0; i < k; i++) {
                        y.set(i, y.doubleValue(i) - h.doubleValue(k, i) * yk);
                    }
                } else {
                    y.set(k, ZERO);
                }
            }

            // Update solution: x = x + sum(y[i] * v[i])
            for (int i = 0; i <= krylovDim; i++) {
                GMRESSolver.axpy(y.doubleValue(i), v[i], x);
            }

            // Compute true residual
            normErr = ZERO;
            for (int i = 0; i < m; i++) {
                Equation row = equations.get(i);
                double ri = row.getRHS() - row.dot(x);
                r.set(row.index, ri);
                normErr = HYPOT.invoke(normErr, ri);
            }

            if (this.isDebugPrinterSet()) {
                this.debug(nbIterations, normErr / normRHS, x);
            }

            if (nbIterations >= iterationsLimit || Double.isNaN(normErr)) {
                break;
            }
        }

        return accuracy.isZero(normRHS) ? normErr : normErr / normRHS;
    }

}
