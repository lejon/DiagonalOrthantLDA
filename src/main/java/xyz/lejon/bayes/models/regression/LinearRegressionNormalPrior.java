package xyz.lejon.bayes.models.regression;

import static xyz.lejon.utils.MatrixOps.rnorm;

import java.io.IOException;

import xyz.lejon.configuration.OLSConfiguration;
import xyz.lejon.utils.MatrixOps;

/**
 * Multiple Bayesian Linear Regression with normal prior on the betas 
 * and inverse gamma prior on sigma
 * 
 * @author Leif Jonsson
 *
 */
public abstract class LinearRegressionNormalPrior extends LinearRegression {
	
	MatrixOps mo = new MatrixOps();
	double SigmaSq = 1.0;
	double SigmaSq0 = 1.0;
	protected double v0 = 1.0;
	protected double tau0 = 1.0;

	public LinearRegressionNormalPrior(OLSConfiguration config, double[][] xs, double[] ys) throws IOException {
		super(config, xs, ys);
		for (int j = 0; j < noCovariates; j++) {
			betas[j] = rnorm(0.0, 1.0);
		}
	}
	
	public abstract double sampleSigmaSquared(Object mu, Object Lambda);
	
	@Override
	public abstract void sampleBeta();
}
