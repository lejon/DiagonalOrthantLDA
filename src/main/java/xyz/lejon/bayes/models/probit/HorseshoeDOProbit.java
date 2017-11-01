package xyz.lejon.bayes.models.probit;

import static java.lang.Math.pow;
import static java.lang.Math.sqrt;
import static xyz.lejon.sampling.BasicRDists.rtexp;
import static xyz.lejon.sampling.BasicRDists.runif;
import static xyz.lejon.utils.MatrixOps.rnorm;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import xyz.lejon.configuration.DOConfiguration;
import xyz.lejon.sampling.BasicRDists;
import xyz.lejon.utils.MatrixOps;

public abstract class HorseshoeDOProbit extends MultivariateParallelDOSampler {
//public abstract class HorseshoeDOProbit extends SerialDOSampler {

	MatrixOps mo = new MatrixOps();
	double [] Tau;
	double [] Sigma;
	double [][] Lambda;
	double c = 100;
	
	static List<double []> sampledLambdas = new ArrayList<double []>();
	static List<Double> sampledTaus       = new ArrayList<Double>();

	public HorseshoeDOProbit(DOConfiguration config, double[][] xs, int[] ys, int noClasses) throws IOException {
		super(config, xs, ys, noClasses);
		Lambda = new double[noClasses][];
		for (int i = 0; i < Lambda.length; i++) {
			Lambda[i] = new double[noCovariates];
			Arrays.fill(Lambda[i], 1.0); // I guess this is tarting value of sampler. Maybe this should be moved to config?
		}
		Tau = new double[noClasses];
		Arrays.fill(Tau, 1.0);
		Sigma = new double[noClasses];
		Arrays.fill(Sigma, 1.0);
		for (int i = 0; i < noClasses; i++) {
			for (int j = 0; j < noCovariates; j++) {
				betas[i][j] = rnorm(0.0, 1.0);
			}
		}
		//printBeta = true;
		System.out.println("################# Running HORSESHOE #################");
	}
	
	/*
	 * This method uses truncated gamma for the draws, but since we have no good implementation
	 * of the trucated gamma, THIS METHOD SHOULD NOT BE USED
	 * @param Tau
	 * @param beta
	 * @param Lambda
	 * @return
	 */
	/*public static double sampleTauTruncatedGamma(double Tau, double [] beta, double [] Lambda) {
		double p = beta.length;
		double gamma = 1 / pow(Tau,2);
		double muHatSq = 0.0;
		for (int i = 0; i < p; i++) {
			double powmu = pow(beta[i] / Lambda[i],2);
			System.out.println("powmu: " + powmu);
			muHatSq += powmu;
		}
		muHatSq /= 2;
		double u = runif(0,1/(1+gamma));
		double uj = (1/u) - 1; // Can there become numerical problems with this or is ((1-u)/u) better?
		System.out.println("Args: " + ((p+1)/2) + " " + muHatSq + " "  + uj);
		double etaHatiNew = rtrgamma((p+1)/2, muHatSq, uj); // Is the arguments correct/parametrisation of the truncated gamma?
		System.out.println("Got gamma draw: " + etaHatiNew);
		double newTau = sqrt(1.0 / etaHatiNew);
		System.out.println("Tau => : " + newTau);
		if(Double.isNaN(etaHatiNew) || Double.isInfinite(newTau)) {
			throw new IllegalStateException("Invalid Tau sampled: " + newTau);
		}
		return newTau;
	}*/

	/**
	 * This method samples Tau given beta, lambda and sigma
	 * 
	 * WARNING: THIS METHOD IS NOT THREAD SAFE, YOU MUST MANUALLY SYNCHRONIZE ACCESS TO THIS METHOD!!
	 * This is due to its use of the native RMath-Java to sample from the truncated gamma distribution.
	 * RMath-Java is not thread safe!
	 * 
	 * If haveIntercept is true, the first element of the beta array will be ignored since, in general
	 * in Bayesian learning we don't regularize the intercept. 
	 * 
	 * @param Tau
	 * @param beta
	 * @param Lambda
	 * @param Sigma
	 * @param haveIntercept
	 * @return
	 */
	public static double sampleTau(double Tau, double [] beta, double [] Lambda, double Sigma, boolean haveIntercept) {
		int trycnt = 0;
		double etaHati = 1 / pow(Tau,2);
		double ub = 1/(1+etaHati);
		double ui = runif(0,ub);
		double muHati = 0.0;
		int i = haveIntercept ? 1 : 0;
		for (; i < beta.length; i++) {
			muHati += pow(beta[i] / (Sigma * Lambda[i]),2);
		}
		muHati /= 2;
		double noPs = haveIntercept ? (double) beta.length - 1 : (double) beta.length;
		double shape = (noPs+1)/2.0;
		double bound = (1.0-ui)/ui; 
		double scale = 1/muHati; 
		double upper = BasicRDists.pgamma(bound, shape, scale, true, false);
		//double eps = Double.MIN_VALUE*1_000_000;
		// double eps = 1.0E-30;
		//upper = upper < eps ? eps : upper;
		double u = runif(0,upper);
		//u = u < eps ? eps : u;
		double etaHatiNew = BasicRDists.qgamma(u, shape, scale);
		
		if(etaHatiNew==0.0) {
			System.err.println("Warning sampleTau: Sampled new eta = 0.0");
			while(trycnt < 5 && etaHatiNew==0.0) {
				etaHatiNew = rtexp(pow(muHati,2)/2, ((1-ui)/ui));
				trycnt++;
			}
			
			if(etaHatiNew==0.0) {
				System.err.println("Warning sampleTau: Gave up after " + trycnt + " tries, setting eta new to 1e-15");
				etaHatiNew = 1.0e-15;
			}
		}
		if(etaHatiNew==0.0 || Double.isNaN(etaHatiNew) || Double.isInfinite(etaHatiNew)) {
			throw new IllegalStateException(buildTauErrorString(Tau, beta, ub, ui, muHati, shape, bound, scale, upper, u, etaHatiNew, Lambda));
		}
		double newTau = 1 / sqrt(etaHatiNew);
		sampledTaus.add(newTau);
		return newTau;
	}
	
	static String buildTauErrorString(double Tau, double[] beta, double ub, double ui, double muHati, double a,
			double bound, double scale, double upper, double u, double etaHatiNew, double [] lambda) {
		return "Invalid etaHatiNew sampled:\n etaHatiNew=" + etaHatiNew
				+ " Tau = " + Tau
				+ " muHati=" + muHati
				+ " a=" + a
				+ " u=" + u
				+ " ub=" + ub
				+ " ui=" + ui
				+ " upper=" + upper
				+ " bound=" + bound
				+ " scale=" + scale + "\n"
				+ MatrixOps.arrToStr(beta, " betas") + "\n"
				+ MatrixOps.arrToStr(lambda, " lambda");
	}

	/*
	public static double [] sampleLambdaInverseQuantile(double Tau, double [] beta, double [] Lambda, double Sigma) {
		double [] newLambdas = new double[Lambda.length];
		for (int i = 0; i < Lambda.length; i++) {
			double gammai = 1 / pow(Lambda[i],2);
			double u1 = runif(0, 1.0 / (1 + gammai));
			double trunc_limit = (1 - u1) / u1;
			double mu2_j = pow(beta[i] / (Tau*Sigma),2);
			double rate_lambda = (mu2_j / 2);
			double ub_lambda = BasicRDists.pexp(trunc_limit, rate_lambda);
			double u2 = runif(0, ub_lambda);
			double gammaNew = BasicRDists.qexp(u2, rate_lambda);
			double lambdaNew = 1 / sqrt(gammaNew);
			if(Double.isNaN(gammaNew) || Double.isInfinite(lambdaNew)) {
				throw new IllegalStateException(buildLambdaErrorString(Tau, beta, Lambda, i, mu2_j, gammaNew, lambdaNew));
			}
			newLambdas[i] = lambdaNew;
		}
		sampledLambdas.add(newLambdas);
		return newLambdas;
	}*/

	/**
	 * Gibbs samples lamda given betas, Tau the previous Lambdas and Sigma
	 * 
	 * @param Tau
	 * @param beta
	 * @param Lambda
	 * @param Sigma
	 * @return
	 */
	public static double [] sampleLambda(double Tau, double [] beta, double [] Lambda, double Sigma) {
		int trycnt = 0;
		double [] newLambdas = new double[Lambda.length];
		for (int i = 0; i < Lambda.length; i++) {
			double gammai = 1 / pow(Lambda[i],2);
			double muHati = beta[i] / (Tau*Sigma);
			double ui = runif(0,1/(1+gammai));
			double gammaNew = rtexp(pow(muHati,2)/2, ((1-ui)/ui)); 
			if(gammaNew==0.0) {
				System.err.println("Warning sampleLambda: Sampled new gamma = 0.0");
				while(trycnt < 5 && gammaNew==0.0) {
					gammaNew = rtexp(pow(muHati,2)/2, ((1-ui)/ui));
					trycnt++;
				}
				
				if(gammaNew==0.0) {
					System.err.println("Warning sampleLambda: Gave up after " + trycnt + " tries, setting gamma new to 1e-15");
					gammaNew = 1.0e-15;
				}
				
			}
			if(Double.isNaN(gammaNew) || gammaNew==0.0) {
				throw new IllegalStateException(buildLambdaErrorString(Tau, beta, Lambda, i, muHati, gammaNew, Double.POSITIVE_INFINITY));
			}
			double lambdaNew = 1 / sqrt(gammaNew); 
			newLambdas[i] = lambdaNew;
		}
		sampledLambdas.add(newLambdas);
		return newLambdas;
	}
	
	static String buildLambdaErrorString(double Tau, double[] beta, double[] Lambda, int i, double muHati,
			double gammaNew, double lambdaNew) {
		return "Invalid Lambda sampled: " + lambdaNew 
				+ " Tau = " + Tau
				+ " Lambda[" + i + "]=" + Lambda[i]
				+ " gammaNew=" + gammaNew
				+ " muHati=" + muHati 
				+ MatrixOps.arrToStr(beta, " Betas");
	}
	
	public void printSampledTaus(int k, int limit) {
		int pCnt = 0;
		System.out.println("Sampled Taus are: ");
		int i = 0;
		int sampleCnt = 0;
		double [] tv = new double[k];
		for(Double t : sampledTaus) {
			tv[i++] = t;
			if(pCnt%k==0) {
				boolean doPrint = (sampledTaus.size()/k)-sampleCnt<limit;
				if(doPrint)
					System.out.print(MatrixOps.arrToStr(tv, "[" + sampleCnt + "] Tau "));
				sampleCnt++;
				tv = new double[k];
				i = 0;
				if(doPrint) System.out.println();
			}
			pCnt++;
		}
	}

	public void printTauMeans(int k) {
		int i = 0;
		double [][] tv = new double[sampledTaus.size()][k];
		int row = 0;
		for(Double t : sampledTaus) {
			tv[row][i++] = t;
			if(i%k==0) {
				row++;
				i = 0;
			}
		}
		System.out.println("Sampled Taus Means are: " + MatrixOps.arrToStr(MatrixOps.colMeans(tv)));
	}

	public void printSampledLambdas(int k, int limit) {
		int pCnt = 0;
		int sampleCnt = 0;
		System.out.println("Sampled Lambdas are: ");
		for(double [] t : sampledLambdas) {
			boolean doPrint = (sampledLambdas.size()/k)-sampleCnt<limit;
			if(doPrint)
				System.out.println(MatrixOps.arrToStr(t, "[" + sampleCnt + "] Lambda "));
			if(pCnt%k==0) {
				if(doPrint) System.out.println();
				sampleCnt++;
			}
			pCnt++;
		}
	}
	
	public void printLambdaMeans(int k) {
		int row = 0;
		int ci = 0;
		double [][][] lambdas = new double[k][sampledLambdas.size()/k][];
		for(double [] t : sampledLambdas) {
			lambdas[ci++][row] = t;
			if(ci%k==0) {
				row++;
				ci = 0;
			}
		}
		
		for (int i = 0; i < k; i++) {			
			System.out.println(i + ": Sampled Lambda Means are: " + MatrixOps.arrToStr(MatrixOps.colMeans(lambdas[i])));
		}
	}

	@Override
	public abstract void sampleBeta(int k);
	
	@Override
	public void postSample() {
		super.postSample();
		printTauMeans(noClasses);
		printLambdaMeans(noClasses);
	}

}
