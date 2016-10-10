package xyz.lejon.sampling;

import java.util.Random;

import org.jblas.DoubleMatrix;

import xyz.lejon.utils.BlasOps;

public class FastMVNSamplerJBLASCholesky {

	protected Random random = new Random();
	protected DoubleMatrix mean;
	protected DoubleMatrix cov;
	protected DoubleMatrix samplingMatrix;
	protected DoubleMatrix eigenVectors;
	protected DoubleMatrix eigenVals;
	
	public FastMVNSamplerJBLASCholesky(final double[] means, final double[][] covariances) {
		mean = new DoubleMatrix(means.length,1);
		for (int row = 0; row < means.length; row++) {
			mean.put(row, 0, means[row]);
		}
		cov = new DoubleMatrix(covariances);
		buildSamplingMatrix();
	}
	
	public FastMVNSamplerJBLASCholesky(DoubleMatrix means, DoubleMatrix covariances) {
		mean = new DoubleMatrix();
		mean.copy(means);
		cov = new DoubleMatrix();
		cov.copy(covariances);

		buildSamplingMatrix();
	}
	
	public FastMVNSamplerJBLASCholesky(DoubleMatrix means, DoubleMatrix invCovariances, boolean useInverse) {
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

	public FastMVNSamplerJBLASCholesky(double [] means, double [][] invCovariances, boolean useInverse) {
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

	public FastMVNSamplerJBLASCholesky(DoubleMatrix covariances) {
		cov = new DoubleMatrix();
		cov.copy(covariances);

		buildSamplingMatrix();
	}
	
	public FastMVNSamplerJBLASCholesky(DoubleMatrix covariances, boolean useInverse) {
		cov = new DoubleMatrix();
		cov.copy(covariances);

		if(useInverse) {			
			buildInverseSamplingMatrix();
		} else {
			buildSamplingMatrix();
		}
	}

	synchronized void buildSamplingMatrix() {
		samplingMatrix = org.jblas.Decompose.cholesky(cov);
	}
	
	/**
	 * This method builds the sampling matrix from the inverse of
	 * the covariance matrix. Invert after decomposition since
	 * this will in general be more stable
	 */
	synchronized void buildInverseSamplingMatrix() {
		DoubleMatrix upper = org.jblas.Decompose.cholesky(cov);
		samplingMatrix = BlasOps.blasInvert(upper);
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
