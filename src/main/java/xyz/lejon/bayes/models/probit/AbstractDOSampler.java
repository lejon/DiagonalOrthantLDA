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
	protected int iterinter = 10;
	private int noSampledBeta;
	boolean useIntecept = false;
	
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
				if(iter % iterinter == 0) {
					System.out.println("Iter: " + iter);
					if(printBeta) {
						System.out.println("Betas: " + MatrixOps.doubleArrayToPrintString(betas,5,5,20));
						System.out.println();
					}
				}	
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
			}
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

	public abstract void sampleBeta(int k); // ??
	
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

}
