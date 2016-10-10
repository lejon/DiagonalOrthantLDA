package xyz.lejon.sampling;

import static java.lang.Math.*; 

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

public class TruncatedNormal {
	
	Random random = new Random();
	TruncatedExponential texp = new TruncatedExponential(1.0);

	// From: https://github.com/olmjo/RcppTN/blob/master/pkg/src/rtn1.cpp
	// This is a port of Jonathan Olmsted's C++ code that implements 
	// an Accept/Reject sampler for a single
	// Truncated Normal random variable with a mixture of algorithms
	// depending on distributional parameters.
	/// Check if simpler sub-algorithm is appropriate.
	protected boolean CheckSimple( double low, ///< lower bound of distribution
			double high ///< upper bound of distribution
			) {
		// Init Values Used in Inequality of Interest
		double val1 = (2 * sqrt(exp(1))) / (low + sqrt(pow(low, 2) + 4));
		double val2 = exp((pow(low, 2) - low * sqrt(pow(low, 2) + 4)) / (4)) ;
		//
		// Test if Simple is Preferred
		if (high > low + val1 * val2) {
			return true ;
		} else {
			return false ;
		}
	}
	/// Draw using algorithm 1.
	///
	/// Naive Accept-Reject algorithm.
	///
	double UseAlg1( double low, ///< lower bound of distribution
			double high ///< upper bound of distribution
			) {
		// Init Valid Flag
		int valid = 0 ;
		//
		// Init Draw Storage
		double z = 0.0 ;
		//
		// Loop Until Valid Draw
		while (valid == 0) {
			//z = random.nextGaussian();
			z = ThreadLocalRandom.current().nextGaussian();
			if (z <= high && z >= low) {
				valid = 1 ;
			}
		}
		//
		// Returns
		return z ;
		//
	}
	
	/// Draw using algorithm 2.
	///
	/// Accept-Reject Algorithm
	///
	///< lower bound of distribution
	double UseAlg2(double low) {
		return texp.draw(low);
	}
	/// Draw using algorithm 3.
	///
	/// Accept-Reject Algorithm
	///
	double UseAlg3( double low, ///< lower bound of distribution
			double high ///< upper bound of distribution
			) {
		// Init Valid Flag
		int valid = 0 ;
		//
		// Declare Qtys
		double rho = 0 ;
		double z = 0 ;
		double u = 0 ;
		//
		// Loop Until Valid Draw
		while (valid == 0) {
			z = randDoubleOnce(low, high);
			if (0 < low) {
				rho = exp((pow(low, 2) - pow(z, 2)) / 2) ;
			} else if (high < 0) {
				rho = exp((pow(high, 2) - pow(z, 2)) / 2) ;
			} else if (0 < high && low < 0) {
				rho = exp(- pow(z, 2) / 2) ;
			}
			u =  random.nextDouble();;
			if (u <= rho) {
				valid = 1 ;
			}
		}
		//
		// Returns
		return z ;
		//
	}
	
	public static double randDoubleOnce(double min, double max) {
	    return ThreadLocalRandom.current().nextDouble() * (max - min) + min;
	}

	/// Draw from an arbitrary truncated normal distribution.
	///
	/// See Robert (1995): <br />
	/// Reference Type: Journal Article <br />
	/// Author: Robert, Christian P. <br />
	/// Primary Title: Simulation of truncated normal variables <br />
	/// Journal Name: Statistics and Computing <br />
	/// Cover Date: 1995-06-01 <br />
	/// Publisher: Springer Netherlands <br />
	/// Issn: 0960-3174 <br />
	/// Subject: Mathematics and Statistics <br />
	// Start Page: 121 <br />
	// End Page: 125 <br />
	/// Volume: 5 <br />
	/// Issue: 2 <br />
	/// Url: http://dx.doi.org/10.1007/BF00143942 <br />
	/// Doi: 10.1007/BF00143942 <br />
	///
	public double rand( double mean,
			double sd,
			double low,
			double high
			) {
		//
		// Init Useful Values
		double draw = 0;
		int type = 0 ;
		int valid = 0 ; // used only when switching to a simplified version
		// of Alg 2 within Type 4 instead of the less
		// efficient Alg 3
		//
		// Set Current Distributional Parameters
		double c_mean = mean ;
		double c_sd = sd ;
		double c_low = low ;
		double c_high = high ;
		double c_stdlow = (c_low - c_mean) / c_sd ;
		double c_stdhigh = (c_high - c_mean) / c_sd ; // bounds are standardized
		//
		// Map Conceptual Cases to Algorithm Cases
		// Case 1 (Simple Deterministic AR)
		// mu \in [low, high]
		if (0 <= c_stdhigh &&
				0 >= c_stdlow
				) {
			type = 1 ;
		}
		// Case 2 (Robert 2009 AR)
		// mu < low, high = Inf
		if (0 < c_stdlow &&
				c_stdhigh == Double.POSITIVE_INFINITY
				) {
			type = 2 ;
		}
		// Case 3 (Robert 2009 AR)
		// high < mu, low = -Inf
		if (0 > c_stdhigh &&
				c_stdlow == Double.NEGATIVE_INFINITY
				) {
			type = 3 ;
		}
		// Case 4 (Robert 2009 AR)
		// mu -\in [low, high] & (abs(low) =\= Inf =\= high)
		if ((0 > c_stdhigh || 0 < c_stdlow) &&
				!(c_stdhigh == Double.POSITIVE_INFINITY || c_stdlow == Double.NEGATIVE_INFINITY)
				) {
			type = 4 ;
		}
		////////////
		// Type 1 //
		////////////
		if (type == 1) {
			draw = UseAlg1(c_stdlow, c_stdhigh) ;
		}
		////////////
		// Type 3 //
		////////////
		if (type == 3) {
			c_stdlow = -1 * c_stdhigh ;
			c_stdhigh = Double.POSITIVE_INFINITY ;
			c_sd = -1 * c_sd ; // hack to get two negative signs to cancel out
			// Use Algorithm #2 Post-Adjustments
			type = 2 ;
		}
		////////////
		// Type 2 //
		////////////
		if (type == 2) {
			draw = UseAlg2(c_stdlow) ;
		}
		////////////
		// Type 4 //
		////////////
		if (type == 4) {
			if (CheckSimple(c_stdlow, c_stdhigh)) {
				while (valid == 0) {
					draw = UseAlg2(c_stdlow) ;
					// use the simple
					// algorithm if it is more
					// efficient
					if (draw <= c_stdhigh) {
						valid = 1 ;
					}
				}
			} else {
				draw = UseAlg3(c_stdlow, c_stdhigh) ; // use the complex
				// algorithm if the simple
				// is less efficient
			}
		}
		// Returns
		return c_mean + c_sd * draw ;
		//
	}

}
