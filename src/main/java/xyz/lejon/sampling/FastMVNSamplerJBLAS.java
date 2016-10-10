package xyz.lejon.sampling;

import java.util.Random;

import org.jblas.DoubleMatrix;
import org.jblas.Eigen;

import xyz.lejon.utils.MatrixOps;

public class FastMVNSamplerJBLAS {

	protected Random random = new Random();
	protected DoubleMatrix mean;
	protected DoubleMatrix cov;
	protected DoubleMatrix samplingMatrix;
	protected DoubleMatrix eigenVectors;
	protected DoubleMatrix eigenVals;
	
	public FastMVNSamplerJBLAS(final double[] means, final double[][] covariances) {
		mean = new DoubleMatrix(means.length,1);
		for (int row = 0; row < means.length; row++) {
			mean.put(row, 0, means[row]);
		}
		cov = new DoubleMatrix(covariances);
		buildSamplingMatrix();
	}
	
	public FastMVNSamplerJBLAS(DoubleMatrix means, DoubleMatrix covariances) {
		mean = new DoubleMatrix();
		mean.copy(means);
		cov = new DoubleMatrix();
		cov.copy(covariances);

		buildSamplingMatrix();
	}
	
	public FastMVNSamplerJBLAS(DoubleMatrix means, DoubleMatrix invCovariances, boolean useInverse) {
		mean = new DoubleMatrix();
		mean.copy(means);
		cov = new DoubleMatrix();
		cov.copy(invCovariances);

		if(useInverse) {			
			buildInverseSamplingMatrix();
		} else {
			buildSamplingMatrix();
		}
	}

	public FastMVNSamplerJBLAS(double [] means, double [][] invCovariances, boolean useInverse) {
		mean = new DoubleMatrix();
		mean.copy(new DoubleMatrix(means));
		cov = new DoubleMatrix();
		cov.copy(new DoubleMatrix(invCovariances));

		if(useInverse) {			
			buildInverseSamplingMatrix();
		} else {
			buildSamplingMatrix();
		}
	}

	public FastMVNSamplerJBLAS(DoubleMatrix covariances) {
		cov = new DoubleMatrix();
		cov.copy(covariances);

		buildSamplingMatrix();
	}
	
	public FastMVNSamplerJBLAS(DoubleMatrix covariances, boolean useInverse) {
		cov = new DoubleMatrix();
		cov.copy(covariances);

		if(useInverse) {			
			buildInverseSamplingMatrix();
		} else {
			buildSamplingMatrix();
		}
	}

	synchronized void buildSamplingMatrix() {
		DoubleMatrix [] evd = Eigen.symmetricEigenvectors(cov);
		eigenVectors = evd[0];
		eigenVals = evd[1];
		DoubleMatrix localEingen = new DoubleMatrix();
		localEingen.copy(eigenVals);
		for (int i = 0; i < localEingen.rows; i++) {
			localEingen.put(i, i, Math.sqrt(localEingen.get(i, i)));
			if(Double.isNaN(localEingen.get(i, i))) {
				throw new IllegalArgumentException("Cannot build sampling matrix, covariance matrix is: " + MatrixOps.doubleArrayToPrintString(cov.toArray2(), 5, 5, 10));
			}
		}

		samplingMatrix = eigenVectors.mmul(localEingen);
	}
	
	/**
	 * This method builds the sampling matrix from the inverse of
	 * the covariance matrix. If the covariance matrix is A then
	 * the it has the same eigenvectors as A^-1 and A^-1 eigenvalues
	 * are the reciprocals of A's 
	 */
	synchronized void buildInverseSamplingMatrix() {
		DoubleMatrix [] evd = Eigen.symmetricEigenvectors(cov);
		// The eigenvectors are the same
		eigenVectors = evd[0];
		eigenVals = evd[1];
		DoubleMatrix localEingen = new DoubleMatrix();
		localEingen.copy(eigenVals);
		for (int i = 0; i < localEingen.rows; i++) {
			// Here we use the reciprocals of the eigenvalues
			localEingen.put(i, i, Math.sqrt(1/localEingen.get(i, i))); 
		}
		samplingMatrix = eigenVectors.mmul(localEingen);
	}

	public synchronized double [] sample() {
		final int dim = mean.rows;
		DoubleMatrix normals = new DoubleMatrix(dim,1);
		for (int i = 0; i < dim; i++) {
			normals.put(i,0, random.nextGaussian());
		}
		DoubleMatrix newMean = samplingMatrix.mmul(normals);
		DoubleMatrix result = newMean.add(mean);
		return result.toArray();
	}
	
	public synchronized double [] sample(DoubleMatrix mean) {
		final int dim = mean.rows;
		DoubleMatrix normals = new DoubleMatrix(dim,1);
		for (int i = 0; i < dim; i++) {
			normals.put(i,0, random.nextGaussian());
		}
		DoubleMatrix newMean = samplingMatrix.mmul(normals);
		DoubleMatrix result = newMean.add(mean);
		return result.toArray();
	}
}
