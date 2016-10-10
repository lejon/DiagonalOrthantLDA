package xyz.lejon.bayes.models.regression;

import java.io.IOException;
import java.util.Arrays;

import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import xyz.lejon.bayes.models.probit.HorseshoeDOProbit;
import xyz.lejon.configuration.OLSConfiguration;
import xyz.lejon.sampling.BasicRDists;
import xyz.lejon.sampling.FastMVNSamplerJBLAS;
import xyz.lejon.sampling.InverseGammaDistribution;
import xyz.lejon.utils.BlasOps;
import xyz.lejon.utils.MatrixOps;

/**
 * Multiple Bayesian Linear Regression with a horseshoe prior on the betas 
 * 
 * @author Leif Jonsson
 *
 */
public class LinearRegressionJBlasHSPrior extends LinearRegressionHSPrior {
	protected DoubleMatrix Xd;
	protected DoubleMatrix XtX;
	protected DoubleMatrix muTilde;
	protected DoubleMatrix ys;
	protected DoubleMatrix Xdt;
	protected DoubleMatrix yty;
	protected DoubleMatrix myPrecision;
	
	public LinearRegressionJBlasHSPrior(OLSConfiguration config, double[][] xs, double[] ys) throws IOException {
		super(config, xs, ys);
		Lambda = new double[noCovariates];
		Arrays.fill(Lambda, 1.0); 
		for (int j = 0; j < noCovariates; j++) {
			betas[j] = BasicRDists.rnorm(0.0, 1.0);
		}
		
		this.ys = new DoubleMatrix(ys);
		
		Xd = new DoubleMatrix(xs);
		Xdt = (new DoubleMatrix(xs)).transpose();
		XtX = Xdt.mmul(Xd);
		yty = this.ys.transpose().mmul(this.ys);

		// OLS estimate
		//DoubleMatrix betahat = BlasOps.blasInvert(XtX).mmul(Xdt).mmul(this.ys);
		//betas = betahat.toArray();
		System.out.println("BetaHat = " + MatrixOps.arrToStr(betas));
				
		myPrecision = new DoubleMatrix(XtX.rows,XtX.columns);
		myPrecision.copy(XtX);
		
		System.out.println("################# Running LinearRegressionJBlasHSPrior #################");
	}
	
	public static double sampleSigmaSquared(DoubleMatrix mu, DoubleMatrix precision, DoubleMatrix yty, int noObservations) {
		double vn = v0  + noObservations / (double) 2;

		DoubleMatrix tmp = yty.sub(mu.transpose().mmul(precision.mmul(mu)));
		
		if(tmp.rows!=1 && tmp.columns!=1) {
			throw new IllegalStateException("intermediary has wrong dims!");
		}
		
		double taun = tau0 + tmp.get(0) / 2;
		
		InverseGammaDistribution ig = new InverseGammaDistribution(vn, taun);

		return ig.nextInverseGamma();
	}
	
	@Override
	public double sampleSigmaSquared(Object muIn, Object precisionIn) {
		DoubleMatrix mu = (DoubleMatrix) muIn;
		DoubleMatrix precision = (DoubleMatrix) precisionIn;

		return sampleSigmaSquared(mu,precision,yty,noRows);
	}
	
	@Override
	public void sampleBeta() {
		Tau    = HorseshoeDOProbit.sampleTau(Tau, betas, Lambda, Sigma, useIntercept);
		Lambda = HorseshoeDOProbit.sampleLambda(Tau, betas, Lambda, Sigma);
				
		sampledTaus.add(Tau);
		sampledLambdas.add(Lambda.clone());

		// Rebuild updated precision matrix
		for (int i = 0; i < myPrecision.rows; i++) {
			double update;
			if(i==0 && useIntercept) {
				update = c;
			} else {
				update = (Math.pow(Tau, 2) * Math.pow(Lambda[i],2));
			}
			myPrecision.put(i, i, XtX.get(i, i) + 1 / update);
		}
		
		DoubleMatrix localTilde = Xdt.mmul(ys);
		DoubleMatrix localMu = new DoubleMatrix(Xd.columns, 1);

		localMu = Solve.solve(myPrecision, localTilde);
		DoubleMatrix pInv = BlasOps.blasInvert(myPrecision);
		
		for (int i = 0; i < pInv.rows; i++) {
			for (int j = 0; j < pInv.columns; j++) {				
				if(Double.isNaN(pInv.get(i,i))) {
					System.err.println(
							"Covariance matrix is broken"
									+ " betas=" + MatrixOps.arrToStr(betas) + "\n"
									+ " Tau=" + Tau + "\n"
									+ " Lambda=" + MatrixOps.arrToStr(Lambda) + "\n"
							);
					System.exit(-1);
				}
			}
		}

		betas = sampleBeta(localMu, pInv, Sigma);
		
		Sigma  = Math.sqrt(sampleSigmaSquared(localMu, myPrecision));
	}

	public static double [] sampleBeta(DoubleMatrix localMu, DoubleMatrix pInv, double sigma) {
		FastMVNSamplerJBLAS mvn = new FastMVNSamplerJBLAS(pInv.muli(Math.pow(sigma,2)));
		double [] betas = mvn.sample(localMu);
		for (int i = 0; i < betas.length; i++) {
			if(Double.isNaN(betas[i])) {
				throw new IllegalArgumentException(
						"Sampled invalid beta:"
						+ " betas=" + MatrixOps.arrToStr(betas) + "\n"
						);
			}
		}
		return betas;
	}
	
	public double printSampledTaus(int limit) {
		double tot = 0.0;
		int cnt = 0;
		System.out.println("Sampled Taus are: ");
		for(Double t : sampledTaus) {
			System.out.print(t + ", ");
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
		double [][] sampledLs = printSampledLambdas(10);
		System.out.println();
		System.out.println("Lambda mean=" + MatrixOps.arrToStr(MatrixOps.colMeans(sampledLs)));
		System.out.println("Tau mean=" + tmean);
		System.out.println();
	}

}
