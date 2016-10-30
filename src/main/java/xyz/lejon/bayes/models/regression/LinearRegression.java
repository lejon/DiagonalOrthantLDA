package xyz.lejon.bayes.models.regression;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import xyz.lejon.configuration.OLSConfiguration;
import xyz.lejon.utils.MatrixOps;

/**
 * Abstract superclass for Multiple Bayesian Linear Regression
 * 
 *  There are two concrete implementations of this, one using
 *  normal priors and one using the horseshoe prior
 * 
 * @author Leif Jonsson
 *
 */
public abstract class LinearRegression {
	Random random = new Random();
	protected int noCovariates;
	protected int noRows; // What is this? No of observations?
	protected double [][] xs;
	protected double [] betas;
	protected double [] betaMeans;
	protected double [] ys;
	protected int burnIn = 0;
	protected int lag = 1;
	protected int iterationsToRun;
	protected boolean traceBeta = false;
	protected boolean printBeta = false;
	protected int iterationsRun = 0;
	protected int currentIteration = 0;
	protected String betaTraceFnPrefix = "beta-trace";
	protected int iterinter = 100;
	protected int noSampledBeta = 0;
	protected boolean useIntercept = true;
	protected double c = 100;
	
	List<double []> sampledBetas = new ArrayList<>();

	public LinearRegression(OLSConfiguration config, double[][] xs, double[] ys) {
		setupSampler(config, xs, ys);
	}

	protected void setupSampler(OLSConfiguration config, double[][] xs, double[] ys) {
		lag = config.getLag();
		burnIn = config.getBurnIn();
		useIntercept = config.getUseIntercept();

		noCovariates = xs[0].length;
		noRows = xs.length;

		this.xs = xs;
		this.ys = ys;
		// Working copy
		betas = new double[noCovariates];
		// Final result
		betaMeans = new double[noCovariates];
	}
	
	public void sample(int iterations) {	
		iterationsToRun = iterations;
		iterinter = iterationsToRun / 5;
		for (int iter = 0; iter < iterations; iter++) {
			currentIteration = iter;
			if(iter % iterinter == 0) {
				System.out.println("Iter: " + iter);
				if(printBeta) {
					System.out.println("Betas: " + MatrixOps.arrToStr(betas));
					System.out.println();
				}
			}
			sampleBeta();
			if(currentIteration > (((double)burnIn/100)*iterationsToRun)) {
				if(currentIteration % lag  == 0) {
					for (int beta = 0; beta < betas.length; beta++) {
						betaMeans[beta] += betas[beta];
					}
					sampledBetas.add(Arrays.copyOf(betas, betas.length));
					noSampledBeta++;
				}
			}
			if(traceBeta) {
				logBetas();
			}
		}
		iterationsRun = iterations;
		postSample();
	}
	
	public void postSample() {
		for (int beta = 0; beta < noCovariates; beta++) {
			betaMeans[beta] /= noSampledBeta; 
		}
	}

	public abstract void sampleBeta();

	public double[] getBetas() {
		return betaMeans;
	}
	
	public List<double []> getSampledBetas() {
		return sampledBetas;
	}

	protected void logBetas() {
		for (int j = 0; j < betas.length; j++) {			
			String traceFile = betaTraceFnPrefix + "-" + j + ".csv";
			try(PrintWriter out = new PrintWriter(new BufferedWriter(
					new FileWriter(traceFile, true)))) {
				for (int betai = 0; betai < betas.length; betai++) {				
					out.print(String.format("%.4f",betas[betai]));
					if(betai+1<betas.length) out.print(", ");
				}
				out.println();
			} catch (IOException e) {
				throw new IllegalStateException(e);
			}
		}
	}
}
