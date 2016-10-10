package xyz.lejon.sampling;

import org.apache.commons.math3.distribution.CauchyDistribution;

public class PositiveHalfCauchy extends CauchyDistribution {
	
	private static final long serialVersionUID = 1L;

	public PositiveHalfCauchy() {
		super();
	}

	public PositiveHalfCauchy(double median, double scale) {
		super(median, scale);
	}

	@Override
	public double sample() {
		return Math.abs(super.sample());
	}

	

}
