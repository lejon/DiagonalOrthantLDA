package xyz.lejon.bayes.models.regression;

import static xyz.lejon.utils.MatrixOps.rnorm;

import java.io.IOException;

import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import xyz.lejon.configuration.OLSConfiguration;
import xyz.lejon.sampling.FastMVNSamplerJBLASCholesky;
import xyz.lejon.sampling.InverseGammaDistribution;

/**
 * Multiple Bayesian Linear Regression with normal prior on the betas 
 * and inverse gamma prior on sigma
 * 
 * @author Leif Jonsson
 *
 */
public class LinearRegressionJBlasNormalPrior extends LinearRegressionNormalPrior {
	protected DoubleMatrix Xd;
	protected DoubleMatrix XtX;
	protected DoubleMatrix Xty;
	protected DoubleMatrix muTilde;
	protected DoubleMatrix ys;
	protected DoubleMatrix yty;
	protected DoubleMatrix Xdt;
	protected DoubleMatrix myPrecision;
	protected DoubleMatrix priorPrecision;
	protected DoubleMatrix priorMean;
	Object blasLock = new Object();

	public LinearRegressionJBlasNormalPrior(OLSConfiguration config, double[][] xs, double[] ys) throws IOException {
		super(config, xs, ys);
		for (int j = 0; j < noCovariates; j++) {
			betas[j] = rnorm(0.0, 1.0);
		}

		this.ys = new DoubleMatrix(ys);
		
		yty = this.ys.transpose().mmul(this.ys);

		Xd = new DoubleMatrix(xs);
		Xdt = (new DoubleMatrix(xs)).transpose();
		Xty = Xdt.mmul(this.ys);
		XtX = Xdt.mmul(Xd);
		priorMean = new DoubleMatrix(Xd.columns, 1);
		priorMean.fill(0.0);

		myPrecision = new DoubleMatrix(XtX.rows,XtX.columns);
		// Build precision matrix
		for (int i = 0; i < myPrecision.rows; i++) {
			for (int j = 0; j < myPrecision.columns; j++) {
				myPrecision.put(i, j,  XtX.get(i, j));
			}
		}

		priorPrecision = new DoubleMatrix(XtX.rows,XtX.columns);
		// Build precision matrix
		for (int i = 0; i < myPrecision.rows; i++) {
			priorPrecision.put(i, i,  SigmaSq0);
		}


		System.out.println("################# Running LinearRegressionJBlasNormalPrior #################");
	}
	
	@Override
	public double sampleSigmaSquared(Object muIn, Object LambdaIn) {
		DoubleMatrix mu = (DoubleMatrix) muIn;
		DoubleMatrix Lambda = (DoubleMatrix) LambdaIn;
		
		double vn = v0  + noRows / (double) 2;

		DoubleMatrix tmp = yty.sub(mu.transpose().mmul(Lambda.mmul(mu)));
		
		if(tmp.rows!=1 && tmp.columns!=1) {
			throw new IllegalStateException("intermediary has wrong dims!");
		}
		
		double taun = tau0 + tmp.get(0) / 2;
		
		InverseGammaDistribution ig = new InverseGammaDistribution(vn, taun);

		return ig.nextInverseGamma();
	}
	
	@Override
	public void sampleBeta() {
		// Rebuild precision matrix
		myPrecision = new DoubleMatrix(XtX.rows,XtX.columns);
		myPrecision = myPrecision.copy(XtX);
		myPrecision = myPrecision.add(priorPrecision);

		DoubleMatrix Xtyssq = new DoubleMatrix(Xty.rows,Xty.columns);
		Xtyssq.copy(Xty);
		
		DoubleMatrix localMu = null;
		synchronized (blasLock) {
			localMu = Solve.solve(myPrecision,Xtyssq);
		}
		
		FastMVNSamplerJBLASCholesky mvn = new FastMVNSamplerJBLASCholesky(myPrecision.mul(1.0/SigmaSq),true);
		betas = mvn.sample(localMu);
		SigmaSq = sampleSigmaSquared(localMu, myPrecision);
	}
}
