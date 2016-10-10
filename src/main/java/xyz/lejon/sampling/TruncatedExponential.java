package xyz.lejon.sampling;

import java.util.Random;

import static java.lang.Math.*;
import static xyz.lejon.sampling.BasicRDists.*;

import org.apache.commons.math3.distribution.ExponentialDistribution;

public class TruncatedExponential {
	double mean = 1.0;
	ExponentialDistribution ex = new ExponentialDistribution(mean);
	Random random = new Random();

	public TruncatedExponential(double mean) {
		this.mean = mean;
		ex = new ExponentialDistribution(mean);
	}
	
	/**
	 * Inverse CDF of right truncated exponential
	 * 
	 */
	public static double qtexp(double quantile, double rate, double truncation) {
		return -log(1-quantile*(1-exp(-truncation*rate)))/rate;
	}
	
	/**
	 * Draw from right truncated Exponential Distribution
	 * 
	 * @param rate (1/scale)
	 * @param highBound right truncation point
	 * @return right truncated exponential covariate
	 */
	public static double rtexp(double rate, double highBound) { 
		return qtexp(runif(), rate, highBound); 
	}

	public double draw(double lowBound) {
		double alphastar = (lowBound + Math.sqrt(Math.pow(lowBound, 2) + 4.0)) / (2.0) ;
		double alpha = alphastar ;
		double e = 0 ;
		double z = 0 ;
		double rho = -1;
		double u = 0 ;

		while (u > rho) {
			e = ex.sample();
			z = lowBound + e / alpha ;
			rho = Math.exp(-Math.pow(alpha - z, 2) / 2) ;
			u = random.nextDouble();
		}
		return z ;
	}

	public static double drawOnce(double lowBound) {
		double alphastar = (lowBound + Math.sqrt(Math.pow(lowBound, 2) + 4.0)) / (2.0) ;
		double alpha = alphastar ;
		double e = 0 ;
		double z = 0 ;
		double rho = -1 ;
		double u = 0 ;

		ExponentialDistribution ex = new ExponentialDistribution(1.0);
		Random random = new Random();

		while (u > rho) {
			e = ex.sample();
			z = lowBound + e / alpha ;
			rho = Math.exp(-Math.pow(alpha - z, 2) / 2) ;
			u = random.nextDouble();
		}
		return z ;
	}

	public static double drawOnce(double lowBound, double mean) {
		double alphastar = (lowBound + Math.sqrt(Math.pow(lowBound, 2) + 4.0)) / (2.0) ;
		double alpha = alphastar ;
		double e = 0 ;
		double z = 0 ;
		double rho = -1;
		double u = 0;

		ExponentialDistribution ex = new ExponentialDistribution(mean);
		Random random = new Random();

		while (u > rho) {
			e = ex.sample();
			z = lowBound + e / alpha ;
			rho = Math.exp(-Math.pow(alpha - z, 2) / 2) ;
			u = random.nextDouble();
		}
		return z ;
	}

}
