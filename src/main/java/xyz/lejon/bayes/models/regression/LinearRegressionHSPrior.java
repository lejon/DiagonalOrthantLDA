package xyz.lejon.bayes.models.regression;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import xyz.lejon.configuration.OLSConfiguration;
import xyz.lejon.sampling.BasicRDists;
import xyz.lejon.utils.MatrixOps;

/**
 * Multiple Bayesian Linear Regression with a horseshoe prior on the betas 
 * 
 * @author Leif Jonsson
 *
 */
public abstract class LinearRegressionHSPrior extends LinearRegression {

	MatrixOps mo = new MatrixOps();
	double Tau = 1;
	double Sigma = 1;
	public static double v0 = 1.0;
	public static double tau0 = 1.0;
	double [] Lambda;
	
	List<double []> sampledLambdas = new ArrayList<double []>();
	List<Double> sampledTaus       = new ArrayList<Double>();

	public LinearRegressionHSPrior(OLSConfiguration config, double[][] xs, double[] ys) throws IOException {
		super(config, xs, ys);
		Lambda = new double[noCovariates];
		Arrays.fill(Lambda, 1.0); 
		for (int j = 0; j < noCovariates; j++) {
			betas[j] = BasicRDists.rnorm(0.0, 1.0);
		}
	}

	public abstract double sampleSigmaSquared(Object mu, Object precision);
	
	@Override
	public abstract void sampleBeta();

	public double printSampledTaus(int limit) {
		double tot = 0.0;
		int cnt = 0;
		System.out.println("Sampled Taus are: ");
		for(Double t : sampledTaus) {
			if(cnt<limit) System.out.print(t + ", ");
			tot += t;
			cnt++;
		}
		return tot / cnt;
	}

	public double [][] printSampledLambdas(int limit) {
		double [][] res = new double[sampledLambdas.size()][];
		int sampleCnt = 0;
		System.out.println("Sampled Lambdas are: ");
		for(double [] t : sampledLambdas) {
			boolean doPrint = sampledLambdas.size()-sampleCnt<limit;
			res[sampleCnt++] = t;
			if(doPrint)
				System.out.println(MatrixOps.arrToStr(t, "[" + sampleCnt + "] Lambda "));
		}
		return res;
	}
	
	public void postSample() {
		super.postSample();
		double tmean = printSampledTaus(10);
		System.out.println();
		double [][] sampledLs = printSampledLambdas(10);
		System.out.println();
		System.out.println("Lambda mean=" + MatrixOps.arrToStr(MatrixOps.colMeans(sampledLs)));
		System.out.println("Tau mean=" + tmean);
		System.out.println();
	}

}
