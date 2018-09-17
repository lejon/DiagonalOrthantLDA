package xyz.lejon.bayes.models.probit;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.ejml.data.DenseMatrix64F;
import org.jblas.DoubleMatrix;

import cc.mallet.topics.LogState;
import cc.mallet.util.LDAUtils;
import xyz.lejon.configuration.DOConfiguration;
import xyz.lejon.sampling.NegativeTruncatedNormal;
import xyz.lejon.sampling.PositiveTruncatedNormal;
import xyz.lejon.utils.MatrixOps;

public abstract class AbstractDOSampler {

	protected PositiveTruncatedNormal ptn = new PositiveTruncatedNormal();
	protected NegativeTruncatedNormal ntn = new NegativeTruncatedNormal();
	Random random = new Random();
	protected int noClasses;
	protected int noCovariates;
	protected int noRows; // What is this? No of observations?
	protected double [][] zsT; // What is this? Generally I think it is better to use u or a for latent normal?
	protected double [][] xs;
	protected double [][] betas;
	protected double [][] betaMeans;
	protected int [] ys;
	protected int burnIn = 0;
	protected int lag = 1;
	protected int iterationsToRun;
	protected boolean traceBeta = false;
	protected boolean printBeta = false;
	protected int iterationsRun = 0;
	protected int currentIteration = 0;
	protected String betaTraceFnPrefix = "beta-trace";
	protected DenseMatrix64F priorPrecision;
	protected DenseMatrix64F priorMean;
	protected double [][] Stilde;
	protected int iterinter = 100;
	protected int noSampledBeta;
	boolean useIntecept = false;
	boolean logLoglikelihood = false;
	protected String loggingPath;
	
	List<double []> [] sampledBetas;

	public AbstractDOSampler() {
		super(); // What does this mean?
	}
	
	@SuppressWarnings("unchecked")
	protected void setupSampler(DOConfiguration config, double[][] xs, int noClasses) {
		lag = config.getLag();
		burnIn = config.getBurnIn();
		useIntecept = config.getUseIntercept();
		
		noCovariates = xs[0].length;
		noRows = xs.length;
		iterinter = config.getIterationPrintInterval();
		
		loggingPath = config.getLoggingUtil() != null ?
				config.getLoggingUtil().getLogDir().getAbsolutePath() :
				"/tmp/";
		if(loggingPath!=null && loggingPath.length()>0) {
			logLoglikelihood = true;
		}

		// Working copy
		betas = new double[noClasses][noCovariates];
		sampledBetas = new ArrayList[noClasses];
		for (int i = 0; i < sampledBetas.length; i++) {
			sampledBetas[i] = new ArrayList<double []>();
		}
		// Final result
		betaMeans = new double[noClasses][noCovariates];

		zsT  = new double[noClasses][noRows];
		for (int k = 0; k < noClasses; k++) {
			for (int row = 0; row < xs.length; row++) {
				double mean = 0;
				if(k==ys[row]) {
					// Positive Truncated Normal
					zsT[k][row] = ptn.rand(mean, 1);
				} else {
					// Negative Truncated Normal
					zsT[k][row] = ntn.rand(mean, 1);
				}
			}
		}
	}
	
	/* (non-Javadoc)
	 * @see models.DOSampler#sample(int)
	 */
	public void sample(int iterations) {	
		iterationsToRun = iterations;
		
		try {
			for (int iter = 0; iter < iterations; iter++) {
				preIteration(iter);
				currentIteration = iter;
				for (int k = 0; k < noClasses; k++) {
					sampleBeta(k);
					if(currentIteration > (((double)burnIn/100)*iterationsToRun)) {
						if(currentIteration % lag  == 0) {
							for (int beta = 0; beta < betas[k].length; beta++) {
								betaMeans[k][beta] += betas[k][beta];
							}
							sampledBetas[k].add(betas[k]);
							noSampledBeta++;
						}
					}
				}
				for (int row = 0; row < noRows; row++) {
					sampleZ(row); // I guess this is latent normal var? Should we rename to A or U?
				}	
				if(traceBeta) {
					logBetas();
				}
				postIteration(iter);
				if (currentIteration % iterinter == 0) {
					String llString = "";
					if(logLoglikelihood) {
						double logLik = calcDoProbitLogLikelihood(xs, ys, betas);
						llString = "(LL:" + logLik + ")";
						LogState logState = new LogState(logLik, currentIteration, null, loggingPath, null);
						LDAUtils.logLikelihoodToFile(logState);					
					}
					System.out.println("Iter: " + iter + " "+ llString);
					if(printBeta) {
						System.out.println("Betas: " + MatrixOps.doubleArrayToPrintString(betas));
						System.out.println();
					}
				}			}
		} catch (Exception e) {
			postSample();
			throw e;
		}
		iterationsRun = iterations;
		postSample();
	}

	protected void preIteration(int iter) {
		
	}

	protected void postIteration(int iter) {
		
	}

	public abstract double [] sampleBeta(int k); // ??
	
	public void sampleZ(int row) {
		for (int k = 0; k < noClasses; k++) {
			double mean = MatrixOps.dot(xs[row],betas[k]);
			// Sample Z_i,y_i | B_y_i ~ N_+( x_i' * Beta, 1)
			if(k==ys[row]) {
				// Positive Truncated Normal
				zsT[k][row] = ptn.rand(mean, 1);
			} else {
				// Negative Truncated Normal
				zsT[k][row] = ntn.rand(mean, 1);
			}
		}
	}
	
	public void printBetaMeans(int limit) {
		int sampleCnt = 0;
		System.out.println("Sampled Betas are: ");
		for(double [] t : betaMeans) {
			boolean doPrint = betaMeans.length-sampleCnt<limit;
			if(doPrint)
				System.out.println(MatrixOps.arrToStr(t, "[" + sampleCnt++ + "] Beta "));
		}
	}
	
	public void postSample() {
		noSampledBeta /= noClasses;
		if(noSampledBeta==0) {
			System.err.println("WARNING: No sampled betas == 0!");
		} else {
			for (int k = 0; k < noClasses; k++) {
				for (int beta = 0; beta < noCovariates; beta++) {
					betaMeans[k][beta] /= noSampledBeta;
				}
			}
			printBetaMeans(100);
		}
	}

	public double[][] getBetas() {
		return betaMeans;
	}
	
	public List<double []> [] getSampledBetas() {
		return sampledBetas;
	}
	
	// TODO: THIS IS BROKEN!!
	public static double [] getClassProbabilities(double [] xi, double [][] betas) {
		int noClasses = betas.length;
		double [] probs = new double[noClasses];
		NormalDistribution nd = new NormalDistribution();
		double sumAll = 0.0;
		for (int classIdx = 0; classIdx < noClasses; classIdx++) {
			double regCoeff = MatrixOps.dot(xi,betas[classIdx]);
			double cumProb = nd.cumulativeProbability(-regCoeff);
			sumAll += (1-cumProb);
		}
	
		for (int calcIdx = 0; calcIdx < noClasses; calcIdx++) {
			double prodAll = 1.0;
			for (int classIdx = 0; classIdx < noClasses; classIdx++) {
				if(classIdx!=calcIdx)
					prodAll *= (1-nd.cumulativeProbability(-MatrixOps.dot(xi,betas[classIdx])));
			}
	
			probs[calcIdx] = (1-nd.cumulativeProbability(-MatrixOps.dot(xi,betas[calcIdx]))) * prodAll / (sumAll * prodAll);
		}
	
		return probs;
	}
	
	protected void logBetas() {
		for (int j = 0; j < betas.length; j++) {			
			String traceFile = betaTraceFnPrefix + "-" + j + ".csv";
			try(PrintWriter out = new PrintWriter(new BufferedWriter(
					new FileWriter(traceFile, true)))) {
				for (int betai = 0; betai < betas[j].length; betai++) {				
					out.print(String.format("%.4f",betas[j][betai]));
					if(betai+1<betas[0].length) out.print(", ");
				}
				out.println();
			} catch (IOException e) {
				throw new IllegalStateException(e);
			}
		}
	}
	
	public double calcDoProbitLogLikelihood(double[][] xs, int[] ys, double[][] betas) {
		return doProbitLogLikelihood(xs, ys, betas); 
	}
	
	public static double doProbitLogLikelihood(double[][] xs, int[] ys, double[][] betas) {
		jdistlib.Normal nd = new jdistlib.Normal(0,1.0);
		double loglik = 0;

		DoubleMatrix X = new DoubleMatrix(xs);
		DoubleMatrix dBetas = new DoubleMatrix(betas);

		DoubleMatrix dBetasT = dBetas.transpose();
		DoubleMatrix dXBetas = X.mmul(dBetasT);

		double [][] XBeta  =  dXBetas.toArray2();
		
		// Iterate over all observations 
		for (int d = 0; d < XBeta.length; d++){
			double DOloglik1;
			double DOloglik2;

			DOloglik1 = 0;
			// First part: Sum and product over J  (in article)
			for (int l = 0; l < XBeta[d].length; l++){
				// For ALL documents with class 'l'
				if(l == ys[d]) {
					double logVal = negativeInfToMinValue(Math.log(1 - nd.cumulative(-XBeta[d][l])));
					DOloglik1 += logVal;
				// For documents with class != 'l'
				} else {
					double logVal = negativeInfToMinValue(Math.log(nd.cumulative(-XBeta[d][l])));
					DOloglik1 += logVal;
				}
			}
			
			// Second part: Normalizing constant
			double DOlik2 = 0;
			for (int j = 0; j < XBeta[d].length; j++){
				double toExp = 0;
				for (int l = 0; l < XBeta[d].length; l++){
					if(j == l) {
						double logVal = negativeInfToMinValue(Math.log(1 - nd.cumulative(-XBeta[d][l])));
						toExp += logVal;
					} else {
						double logVal = negativeInfToMinValue(Math.log(nd.cumulative(-XBeta[d][l])));
						toExp += logVal;
					}						
				}
				DOlik2 += Math.exp(toExp);
			}
			DOloglik2 = Math.log(DOlik2);
			
			// Divide (i.e subtract log normalizing constant) with normalizing constant to get normalized LL
			loglik += DOloglik1 - DOloglik2;
		}
		return loglik;
	}

	private static double negativeInfToMinValue(double log) {
		if(log==Double.NEGATIVE_INFINITY) return Double.MIN_VALUE;
		return log;
	}

}
