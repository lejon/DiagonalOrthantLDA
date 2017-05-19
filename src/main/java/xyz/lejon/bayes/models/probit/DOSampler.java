package xyz.lejon.bayes.models.probit;

import java.util.List;

public interface DOSampler {
	void sample(int iterations);
	double[][] getBetas();
	List<double[]>[] getSampledBetas();
	double doProbitLikelihood();
}
