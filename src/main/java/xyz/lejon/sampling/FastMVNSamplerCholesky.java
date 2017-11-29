package xyz.lejon.sampling;

import java.util.Random;

import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.DecompositionFactory;
import org.ejml.ops.CommonOps;

import xyz.lejon.utils.MatrixOps;

public class FastMVNSamplerCholesky {

	Random random = new Random();
	DenseMatrix64F mean;
	DenseMatrix64F cov;
	DenseMatrix64F samplingMatrix;

	public FastMVNSamplerCholesky(final double[] means, final double[][] covariances) {
		mean = new DenseMatrix64F(means.length,1);
		for (int row = 0; row < means.length; row++) {
			mean.set(row, 0, means[row]);
		}
		
		cov = new DenseMatrix64F(covariances);

		buildSamplingMatrix();
	}
	
	public FastMVNSamplerCholesky(DenseMatrix64F means, DenseMatrix64F covariances) {
		mean = means.copy();		
		cov = covariances.copy();

		buildSamplingMatrix();
	}

	public FastMVNSamplerCholesky(DenseMatrix64F means, DenseMatrix64F covariances, boolean useInverse) {
		mean = means.copy();		
		cov = covariances.copy();

		if(useInverse) {			
			buildInverseSamplingMatrix();
		} else {
			buildSamplingMatrix();
		}

	}


	public FastMVNSamplerCholesky(final double[] means, final double[][] covariances, boolean useInverse) {
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

	protected void buildSamplingMatrix() {
		org.ejml.interfaces.decomposition.CholeskyDecomposition<DenseMatrix64F> 
			ch = DecompositionFactory.chol(cov.numRows, true);

		if( !ch.decompose(cov) )
			throw new RuntimeException("Decomposition failed");

		DenseMatrix64F T = ch.getT(new DenseMatrix64F(mean.numRows, mean.numRows));
		if(!ch.isLower()) {
			CommonOps.transpose(T);
		}
		samplingMatrix = T;
	}

	protected void buildInverseSamplingMatrix() {
		org.ejml.interfaces.decomposition.CholeskyDecomposition<DenseMatrix64F> 
			ch = DecompositionFactory.chol(cov.numRows, true);

		DenseMatrix64F invCov = cov.copy();
		//CommonOps.invert(invCov);
		
		if( !ch.decompose(invCov) )
			throw new RuntimeException("Decomposition failed");

		DenseMatrix64F T = ch.getT(new DenseMatrix64F(mean.numRows, mean.numRows));
		if(ch.isLower()) {
			CommonOps.transpose(T);
		}
		CommonOps.invert(T);
		samplingMatrix = T;
	}
	
	public double [] sample() {
		final int dim = mean.numRows;
		final DenseMatrix64F normals = new DenseMatrix64F(dim,1);
		final DenseMatrix64F newMean = new DenseMatrix64F(dim,1);
		final DenseMatrix64F result = new DenseMatrix64F(dim,1);

		for (int i = 0; i < dim; i++) {
			normals.set(i,0, random.nextGaussian());
		}

		CommonOps.mult(samplingMatrix, normals, newMean);
		CommonOps.add(mean, newMean, result);
		CommonOps.transpose(result);
		
		double[] ds = MatrixOps.extractDoubleArray(result)[0];
		return ds;
	}
}
