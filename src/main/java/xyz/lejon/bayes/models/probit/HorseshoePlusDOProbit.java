package xyz.lejon.bayes.models.probit;

import static org.ejml.ops.CommonOps.addEquals;
import static org.ejml.ops.CommonOps.invert;
import static org.ejml.ops.CommonOps.multTransA;

import java.io.IOException;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import xyz.lejon.configuration.DOConfiguration;
import xyz.lejon.sampling.PositiveHalfCauchy;
import xyz.lejon.utils.MatrixOps;

public class HorseshoePlusDOProbit extends SerialDOSampler {

	double tau = 0.5;
	// Draw prior on tau from Positive Cauchy ~ (0,1)
	PositiveHalfCauchy phc = new PositiveHalfCauchy(0.0,1);
	double [][] precMtrx;

	public HorseshoePlusDOProbit(DOConfiguration config, double[][] xs, int[] ys) throws IOException {
		super(config, xs, ys);
		precMtrx = new double[Xd.numCols][Xd.numCols];
		
		PositiveHalfCauchy drawTau = new PositiveHalfCauchy(0.0,1.0/noClasses);
		tau = drawTau.sample();
		//Alternative tau ~ U(0,1)	
	}

	@Override
	public double [] sampleBeta(int k) {
		double [] zColk = zsT[k];

		// Resample prior 
		DenseMatrix64F zColKd = new DenseMatrix64F(zColk.length,1);
		zColKd.setData(zColk);
		// Draw local prior lambda from Positive Cauchy ~ (0,1)
		// Draw lambda from Positive Cauchy ~ (0,eta_j*tau)
		double [] lambda = new double [precMtrx.length];
		for (int i = 0; i < lambda.length; i++) {
			double eta_j = phc.sample();
			PositiveHalfCauchy pch = new PositiveHalfCauchy(0.0,eta_j*tau);
			lambda[i] = pch.sample();
		}
		// Rebuild precision matrix
		for (int i = 0; i < precMtrx.length; i++) {
			for (int j = 0; j < precMtrx.length; j++) {
				if(i==j) {
					precMtrx[i][j] = 1 / Math.pow(lambda[i],2);
				}
			}
		}
		priorPrecision = new DenseMatrix64F(precMtrx);

		DenseMatrix64F XtXA = new DenseMatrix64F(XtX);
		addEquals(XtXA, priorPrecision);
		Stilde = MatrixOps.extractDoubleArray(XtXA);
		invert(XtXA);

		multTransA(Xd, zColKd, muTilde);
		CommonOps.mult(XtXA, muTilde, mumu);
		double [] mu_tile = mumu.getData();

		MultivariateNormalDistribution mvn = new MultivariateNormalDistribution(mu_tile, Stilde);

		betas[k] = mvn.sample();
		return betas[k];
	}
}
