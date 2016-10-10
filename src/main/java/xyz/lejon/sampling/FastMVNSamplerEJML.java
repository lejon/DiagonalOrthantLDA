package xyz.lejon.sampling;

import java.util.Random;

import org.apache.commons.math3.util.FastMath;
import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.DecompositionFactory;
import org.ejml.interfaces.decomposition.EigenDecomposition;
import org.ejml.ops.CommonOps;
import org.ejml.ops.EigenOps;

import xyz.lejon.utils.MatrixOps;

public class FastMVNSamplerEJML {

	protected Random random = new Random();
	protected DenseMatrix64F mean;
	protected DenseMatrix64F cov;
	protected DenseMatrix64F samplingMatrix;

	public FastMVNSamplerEJML(final double[] means, final double[][] covariances) {
		mean = new DenseMatrix64F(means.length,1);
		for (int row = 0; row < means.length; row++) {
			mean.set(row, 0, means[row]);
		}
		
		cov = new DenseMatrix64F(covariances);

		buildSamplingMatrix();
	}
	
	public FastMVNSamplerEJML(final double[] means, final double[][] covariances, boolean useInverse) {
		mean = new DenseMatrix64F(means.length,1);
		for (int row = 0; row < means.length; row++) {
			mean.set(row, 0, means[row]);
		}
		
		cov = new DenseMatrix64F(covariances);

		if(useInverse) {			
			buildInverseSamplingMatrix();
		} else {
			buildSamplingMatrix();
		}
	}

	
	public FastMVNSamplerEJML(DenseMatrix64F means, DenseMatrix64F covariances) {
		mean = means.copy();		
		cov = covariances.copy();

		buildSamplingMatrix();
	}

	public FastMVNSamplerEJML(DenseMatrix64F means, DenseMatrix64F covariances, boolean useInverse) {
		mean = means.copy();		
		cov = covariances.copy();

		if(useInverse) {			
			buildInverseSamplingMatrix();
		} else {
			buildSamplingMatrix();
		}
	}
	
	protected synchronized void buildSamplingMatrix() {
		EigenDecomposition<DenseMatrix64F> eig = DecompositionFactory.eig(cov.numRows, true, true);

		if( !eig.decompose(cov) )
			throw new RuntimeException("Decomposition failed");

		DenseMatrix64F eigenVectors = EigenOps.createMatrixV(eig);
		DenseMatrix64F eigenVals = EigenOps.createMatrixD(eig);
		DenseMatrix64F localEigenVals = eigenVals.copy();

		// Scale each eigenvector by the square root of its eigenvalue.
		for (int row = 0; row < localEigenVals.numRows; row++) {
			localEigenVals.set(row, row, FastMath.sqrt(localEigenVals.get(row,row)));
		}
		samplingMatrix = new DenseMatrix64F(eigenVectors.numRows,eigenVectors.numCols);

		CommonOps.mult(eigenVectors, localEigenVals, samplingMatrix);
	}
	
	/**
	 * This method builds the sampling matrix from the inverse of
	 * the covariance matrix. If the covariance matrix is A then
	 * the it has the same eigenvectors as A^-1 and A^-1 eigenvalues
	 * are the reciprocals of A's 
	 */
	protected synchronized void buildInverseSamplingMatrix() {
		EigenDecomposition<DenseMatrix64F> eig = DecompositionFactory.eig(cov.numRows, true, true);

		if( !eig.decompose(cov) )
			throw new RuntimeException("Decomposition failed");

		DenseMatrix64F eigenVectors = EigenOps.createMatrixV(eig);
		DenseMatrix64F eigenVals = EigenOps.createMatrixD(eig);
		DenseMatrix64F localEigenVals = eigenVals.copy();
		
		for (int i = 0; i < localEigenVals.numRows; i++) {
			// Here we use the reciprocals of the eigenvalues
			localEigenVals.set(i, i, Math.sqrt(1/localEigenVals.get(i, i))); 
		}
		
		samplingMatrix = new DenseMatrix64F(eigenVectors.numRows,eigenVectors.numCols);
		
		CommonOps.mult(eigenVectors, localEigenVals, samplingMatrix);
	}

	/**
	 * Generate sample using the mean set during construction
	 * @return
	 */
	public synchronized double [] sample() {
		final int dim = mean.numRows;
		final DenseMatrix64F normals = new DenseMatrix64F(dim,1);
		final DenseMatrix64F newMean = new DenseMatrix64F(dim,1);
		final DenseMatrix64F result = new DenseMatrix64F(dim,1);

		for (int i = 0; i < dim; i++) {
			normals.set(i,0, random.nextGaussian());
		}

		CommonOps.mult(samplingMatrix, normals, newMean);

		for (int i = 0; i < dim; i++) {
			CommonOps.add(mean, newMean, result);
		}

		CommonOps.transpose(result);
		double[] ds = MatrixOps.extractDoubleArray(result)[0];
		return ds;
	}
	
	/**
	 * Generate sample using the given mean and covariance matrix set during construction
	 * 
	 * @param mean
	 * @return
	 */
	public synchronized double [] sample(DenseMatrix64F mean) {
		final int dim = mean.numRows;
		final DenseMatrix64F normals = new DenseMatrix64F(dim,1);
		final DenseMatrix64F newMean = new DenseMatrix64F(dim,1);
		final DenseMatrix64F result = new DenseMatrix64F(dim,1);

		for (int i = 0; i < dim; i++) {
			normals.set(i,0, random.nextGaussian());
		}

		CommonOps.mult(samplingMatrix, normals, newMean);

		for (int i = 0; i < dim; i++) {
			CommonOps.add(mean, newMean, result);
		}

		CommonOps.transpose(result);
		double[] ds = MatrixOps.extractDoubleArray(result)[0];
		return ds;
	}

}
