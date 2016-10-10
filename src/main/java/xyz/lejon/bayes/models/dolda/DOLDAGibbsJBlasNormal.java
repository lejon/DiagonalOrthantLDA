package xyz.lejon.bayes.models.dolda;

import org.apache.commons.configuration.ConfigurationException;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import xyz.lejon.configuration.DOLDAConfiguration;
import xyz.lejon.sampling.FastMVNSamplerJBLAS;
import xyz.lejon.sampling.InverseGammaDistribution;
import xyz.lejon.utils.BlasOps;

/**
 * This class implements the samples the beta coefficients of the DO Probit
 * using oridinary Bayesian Linear Regression (BLR) including sampling 
 * sigma squared
 * 
 * @author Leif Jonsson
 *
 */
public class DOLDAGibbsJBlasNormal extends EDOLDANormal {

	private static final long serialVersionUID = 1L;

	public DOLDAGibbsJBlasNormal(DOLDAConfiguration parentCfg, double[][] xs, int[] iys) throws ConfigurationException {
		super(parentCfg, xs, iys);
		System.out.println("Using Beta sampler: " + this.getClass().getName());
	}
	
	@Override
	protected void sampleBetas(double [][] Xs, double [][] XtXs, double [][] Xts, double [][] precisions) {
		for (int k = 0; k < noClasses; k++) {
			sampleBeta(Xs, XtXs, Xts, precisions, k);
			saveBetaSample(k);
		}
	}
	
	public double sampleSigmaSquared(DoubleMatrix mu, DoubleMatrix Lambda, DoubleMatrix yty) {
		double vn = v0  + ys.length / (double) 2;
	
		DoubleMatrix tmp = null;
		synchronized (betaLock) {
			tmp = yty.sub(mu.transpose().mmul(Lambda.mmul(mu)));
		}
		
		if(tmp.rows!=1 && tmp.columns!=1) {
			throw new IllegalStateException("intermediary has wrong dims!");
		}
		
		double taun = tau0 + tmp.get(0) / 2;
		
		InverseGammaDistribution ig = new InverseGammaDistribution(vn, taun);
	
		return ig.nextInverseGamma();
	}

	protected void sampleBeta(double [][] Xs, double [][] XtXs, double [][] Xts, double [][] precisions,  int k) {		
		DoubleMatrix X = new DoubleMatrix(Xs);
		DoubleMatrix Xt = new DoubleMatrix(Xts);
		DoubleMatrix precision = new DoubleMatrix(precisions);
	
		// Update diagonal elements of precision matrix
		for (int i = 0; i < precision.rows; i++) {
			precision.put(i, i, XtXs[i][i] + (1 / c));
		}

		double [] zColk = Ast[k];
		DoubleMatrix zColKd = new DoubleMatrix(zColk);
		DoubleMatrix localTilde = null;
		DoubleMatrix localMu = null;
		DoubleMatrix pInv = null;

		synchronized (betaLock) {
			localTilde = Xt.mmul(zColKd);
			localMu = new DoubleMatrix(X.columns, 1);

			localMu = Solve.solve(precision, localTilde);
			pInv = BlasOps.blasInvert(precision);
			//localMu = pInv.mmul(localTilde);
		}
		
		FastMVNSamplerJBLAS mvn = new FastMVNSamplerJBLAS(pInv.mul(SigmaSq[k]));
		//FastMVNSamplerJBLAS mvn = new FastMVNSamplerJBLAS(precision, true);
		betas[k] = mvn.sample(localMu);

		DoubleMatrix yty = null;
		synchronized (betaLock) {
			yty = zColKd.transpose().mmul(zColKd);
		}
		// The below method call MUST NOT BE MOVED INTO A SYNCHRONIZED BLOCK, YOU *WILL* GET DEADLOCK!!
		SigmaSq[k] = sampleSigmaSquared(localMu, precision, yty);
	}
}
