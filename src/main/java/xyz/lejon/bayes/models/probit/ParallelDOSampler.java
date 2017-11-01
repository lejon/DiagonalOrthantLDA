package xyz.lejon.bayes.models.probit;
import java.io.IOException;

import org.apache.commons.math3.distribution.NormalDistribution;

import xyz.lejon.configuration.DOConfiguration;
import xyz.lejon.utils.MatrixOps;


public class ParallelDOSampler extends AbstractParallelDOSampler implements DOSampler {
	NormalDistribution nd;
	protected double c = 10;

	public ParallelDOSampler(DOConfiguration config, double [][] xs, int [] ys, int noClasses) throws IOException {
		this.xs = xs;
		this.ys = ys;
		this.noClasses = noClasses;
		setupSampler(config, xs, noClasses);
	}

	public double [] sampleBeta(int k) {
		double [] zColk = zsT[k];
		for (int beta = 0; beta < betas.length; beta++) {	
			System.out.println(currentIteration + ": Sample beta[" + k +"][" + beta + "]");
			double [] xColk = MatrixOps.extractColVector(k,xs);
			
			double mean = MatrixOps.dot(zColk, xColk);
			double stdev = MatrixOps.dot(xColk,xColk);
			
			nd = new NormalDistribution(stdev*mean,c/stdev);
			
			betas[k][beta] = nd.sample();
		}
		return betas[k];
	}
	
	@Override
	public void postSample() {}
}
