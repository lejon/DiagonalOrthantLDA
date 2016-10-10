package xyz.lejon.sampling;

import org.apache.commons.math3.distribution.AbstractMultivariateRealDistribution;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.NonPositiveDefiniteMatrixException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.SynchronizedRandomGenerator;
import org.apache.commons.math3.random.Well19937c;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.MathArrays;

public class FixedCovarianceMultivariateNormalDistribution extends AbstractMultivariateRealDistribution {
	/** Vector of means. */
	private final double[] means;
	/** Covariance matrix. */
	private final RealMatrix covarianceMatrix;
	/** The matrix inverse of the covariance matrix. */
	private RealMatrix covarianceMatrixInverse;
	/** The determinant of the covariance matrix. */
	private final double covarianceMatrixDeterminant;
	/** Matrix used in computation of samples. */
	private final RealMatrix samplingMatrix;

	public enum CoVarianceType { COVARIANCE, PRECISION };
	/**
	 * Creates a multivariate normal distribution with the given mean vector and
	 * covariance matrix.
	 * <br/>
	 * The number of dimensions is equal to the length of the mean vector
	 * and to the number of rows and columns of the covariance matrix.
	 * It is frequently written as "p" in formulae.
	 *
	 * @param means Vector of means.
	 * @param covariances Covariance matrix.
	 * @throws DimensionMismatchException if the arrays length are
	 * inconsistent.
	 * @throws SingularMatrixException if the eigenvalue decomposition cannot
	 * be performed on the provided covariance matrix.
	 * @throws NonPositiveDefiniteMatrixException if any of the eigenvalues is
	 * negative.
	 */
	public FixedCovarianceMultivariateNormalDistribution(final double[] means,
			final double[][] covariances)
					throws SingularMatrixException,
					DimensionMismatchException,
					NonPositiveDefiniteMatrixException {
		this(new SynchronizedRandomGenerator(new Well19937c()), means, covariances);
	}

	/**
	 * Creates a multivariate normal distribution with the given mean vector and
	 * covariance matrix.
	 * <br/>
	 * The number of dimensions is equal to the length of the mean vector
	 * and to the number of rows and columns of the covariance matrix.
	 * It is frequently written as "p" in formulae.
	 *
	 * @param rng Random Number Generator.
	 * @param means Vector of means.
	 * @param covariances Covariance matrix.
	 * @throws DimensionMismatchException if the arrays length are
	 * inconsistent.
	 * @throws SingularMatrixException if the eigenvalue decomposition cannot
	 * be performed on the provided covariance matrix.
	 * @throws NonPositiveDefiniteMatrixException if any of the eigenvalues is
	 * negative.
	 */
	public FixedCovarianceMultivariateNormalDistribution(RandomGenerator rng,
			final double[] means,
			final double[][] covariances)
					throws SingularMatrixException,
					DimensionMismatchException,
					NonPositiveDefiniteMatrixException {
		this(rng,means,covariances,CoVarianceType.COVARIANCE);
	}
	
	/**
	 * Creates a multivariate normal distribution with the given mean vector and
	 * covariance matrix.
	 * <br/>
	 * The number of dimensions is equal to the length of the mean vector
	 * and to the number of rows and columns of the covariance matrix.
	 * It is frequently written as "p" in formulae.
	 *
	 * @param rng Random Number Generator.
	 * @param means Vector of means.
	 * @param covariances Covariance matrix.
	 * @param type are we supplied a covariance or precision matrix
	 * @throws DimensionMismatchException if the arrays length are
	 * inconsistent.
	 * @throws SingularMatrixException if the eigenvalue decomposition cannot
	 * be performed on the provided covariance matrix.
	 * @throws NonPositiveDefiniteMatrixException if any of the eigenvalues is
	 * negative.
	 */
	public FixedCovarianceMultivariateNormalDistribution(RandomGenerator rng,
			final double[] means,
			final double[][] covariances, CoVarianceType type)
					throws SingularMatrixException,
					DimensionMismatchException,
					NonPositiveDefiniteMatrixException {
		super(rng, means.length);

		final int dim = means.length;

		if (covariances.length != dim) {
			throw new DimensionMismatchException(covariances.length, dim);
		}

		for (int i = 0; i < dim; i++) {
			if (dim != covariances[i].length) {
				throw new DimensionMismatchException(covariances[i].length, dim);
			}
		}

		this.means = MathArrays.copyOf(means);

		covarianceMatrix = new Array2DRowRealMatrix(covariances);

		// Covariance matrix eigen decomposition.
		final EigenDecomposition covMatDec = new EigenDecomposition(covarianceMatrix);

		// Compute and store the determinant.
		covarianceMatrixDeterminant = covMatDec.getDeterminant();

		// Eigenvalues of the covariance matrix.
		final double[] covMatEigenvalues = covMatDec.getRealEigenvalues();

		for (int i = 0; i < covMatEigenvalues.length; i++) {
			if (covMatEigenvalues[i] < 0) {
				throw new NonPositiveDefiniteMatrixException(covMatEigenvalues[i], i, 0);
			}
		}

		// Matrix where each column is an eigenvector of the covariance matrix.
		final Array2DRowRealMatrix covMatEigenvectors = new Array2DRowRealMatrix(dim, dim);
		for (int v = 0; v < dim; v++) {
			final double[] evec = covMatDec.getEigenvector(v).toArray();
			covMatEigenvectors.setColumn(v, evec);
		}

		final RealMatrix tmpMatrix = covMatEigenvectors.transpose();

		// Scale each eigenvector by the square root of its eigenvalue.
		for (int row = 0; row < dim; row++) {
			final double factor = FastMath.sqrt(covMatEigenvalues[row]);
			for (int col = 0; col < dim; col++) {
				tmpMatrix.multiplyEntry(row, col, factor);
			}
		}

		samplingMatrix = covMatEigenvectors.multiply(tmpMatrix);
	}

	/**
	 * Gets the mean vector.
	 *
	 * @return the mean vector.
	 */
	public double[] getMeans() {
		return MathArrays.copyOf(means);
	}

	/**
	 * Gets the covariance matrix.
	 *
	 * @return the covariance matrix.
	 */
	public RealMatrix getCovariances() {
		return covarianceMatrix.copy();
	}

	/** {@inheritDoc} */
	public double density(final double[] vals) throws DimensionMismatchException {
		final int dim = getDimension();
		if (vals.length != dim) {
			throw new DimensionMismatchException(vals.length, dim);
		}

		return FastMath.pow(2 * FastMath.PI, -0.5 * dim) *
				FastMath.pow(covarianceMatrixDeterminant, -0.5) *
				getExponentTerm(vals);
	}

	/**
	 * Gets the square root of each element on the diagonal of the covariance
	 * matrix.
	 *
	 * @return the standard deviations.
	 */
	public double[] getStandardDeviations() {
		final int dim = getDimension();
		final double[] std = new double[dim];
		final double[][] s = covarianceMatrix.getData();
		for (int i = 0; i < dim; i++) {
			std[i] = FastMath.sqrt(s[i][i]);
		}
		return std;
	}

	/** {@inheritDoc} */
	@Override
	public double[] sample() {
		final int dim = getDimension();
		final double[] normalVals = new double[dim];

		for (int i = 0; i < dim; i++) {
			normalVals[i] = random.nextGaussian();
		}

		final double[] vals = samplingMatrix.operate(normalVals);

		for (int i = 0; i < dim; i++) {
			vals[i] += means[i];
		}

		return vals;
	}
	
	public double[] sample(double [] means) {
		final int dim = getDimension();
		final double[] normalVals = new double[dim];

		for (int i = 0; i < dim; i++) {
			normalVals[i] = random.nextGaussian();
		}

		final double[] vals = samplingMatrix.operate(normalVals);

		for (int i = 0; i < dim; i++) {
			vals[i] += means[i];
		}

		return vals;
	}

	/**
	 * Computes the term used in the exponent (see definition of the distribution).
	 *
	 * @param values Values at which to compute density.
	 * @return the multiplication factor of density calculations.
	 */
	private double getExponentTerm(final double[] values) {
		final double[] centered = new double[values.length];
		for (int i = 0; i < centered.length; i++) {
			centered[i] = values[i] - getMeans()[i];
		}
		if(covarianceMatrixInverse==null) {
			// Covariance matrix eigen decomposition.
			final EigenDecomposition covMatDec = new EigenDecomposition(covarianceMatrix);
			// Compute and store the inverse.
			covarianceMatrixInverse = covMatDec.getSolver().getInverse();
		}

		final double[] preMultiplied = covarianceMatrixInverse.preMultiply(centered);
		double sum = 0;
		for (int i = 0; i < preMultiplied.length; i++) {
			sum += preMultiplied[i] * centered[i];
		}
		return FastMath.exp(-0.5 * sum);
	}
}
