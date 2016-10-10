package xyz.lejon.sampling;

public interface MVNSampler {

	/**
	 * Generate sample using the mean set during construction
	 * @return
	 */
	double[] sample();

}