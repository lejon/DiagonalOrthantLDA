package xyz.lejon.sampling;


public final class BasicRDists {
	
	public static jdistlib.rng.RandomEngine mt = new SafeMersenneTwister();
	// The MersenneTwister is not thread safe
	//public static jdistlib.rng.RandomEngine mt = new jdistlib.rng.RandomCMWC();
	//public static jdistlib.rng.RandomEngine mt = new JDKRandomEngine();

	/**
	 * 
	 * The Uniform Distribution
	 * Description
	 * 
	 * These functions provide information about the uniform distribution on 
	 * the interval from min to max. dunif gives the density, punif gives the 
	 * distribution function qunif gives the quantile function and runif 
	 * generates random deviates.
	 * 
	 * Usage
	 * 
	 * dunif(x, min = 0, max = 1, log = FALSE)
	 * punif(q, min = 0, max = 1, lower.tail = TRUE, log.p = FALSE)
	 * qunif(p, min = 0, max = 1, lower.tail = TRUE, log.p = FALSE)
	 * runif(n, min = 0, max = 1)
	 * 
	 * Arguments
	 * x, q
	 * 	vector of quantiles.
	 * p	
	 * 	vector of probabilities.
	 * n	
	 * 	number of observations. If length(n) > 1, the length is taken to be the number required.
	 * min, max	
	 * 	lower and upper limits of the distribution. Must be finite.
	 * log, log.p	
	 * 	logical; if TRUE, probabilities p are given as log(p).
	 * lower.tail	
	 * 	logical; if TRUE (default), probabilities are P[X ≤ x], otherwise, P[X > x].
	 * 
	 * Details
	 * 	If min or max are not specified they assume the default values of 0 and 1 respectively.
	 * 
	 * @param a
	 * @param b
	 * @return
	 */
	public static double runif(double a, double b) {
		return jdistlib.Uniform.random(a, b, mt);
	}

	public static double runif() {
		return jdistlib.Uniform.random(0,1,mt);
	}


	/**
	 * 
	 * The Normal Distribution
	 * Description
	 * 
	 * Density, distribution function, quantile function and random generation for the 
	 * normal distribution with mean equal to mean and standard deviation equal to sd.
	 * 
	 * Usage
	 * 
	 * dnorm(x, mean = 0, sd = 1, log = FALSE)
	 * pnorm(q, mean = 0, sd = 1, lower.tail = TRUE, log.p = FALSE)
	 * qnorm(p, mean = 0, sd = 1, lower.tail = TRUE, log.p = FALSE)
	 * rnorm(n, mean = 0, sd = 1)
	 * Arguments
	 * 
	 * x, q	
	 * vector of quantiles.
	 * 
	 * p	
	 * vector of probabilities.
	 * 
	 * n	
	 * number of observations. If length(n) > 1, the length is taken to be the number required.
	 * 
	 * mean	
	 * vector of means.
	 * 
	 * sd	
	 * vector of standard deviations.
	 * 
	 * log, log.p	
	 * logical; if TRUE, probabilities p are given as log(p).
	 * 
	 * lower.tail	
	 * logical; if TRUE (default), probabilities are P[X ≤ x] otherwise, P[X > x].
	 * 
	 * Details
	 * 
	 * If mean or sd are not specified they assume the default values of 0 and 1, respectively.
	 * The normal distribution has density
	 * f(x) = 1/(√(2 π) σ) e^-((x - μ)^2/(2 σ^2))
	 * where μ is the mean of the distribution and σ the standard deviation.
	 * 
	 * Value
	 * dnorm gives the density, pnorm gives the distribution function, qnorm gives the quantile function, and rnorm generates random deviates.
	 * The length of the result is determined by n for rnorm, and is the maximum of the lengths of the numerical arguments for the other functions.
	 * The numerical arguments other than n are recycled to the length of the result. Only the first elements of the logical arguments are used.
	 * For sd = 0 this gives the limit as sd decreases to 0, a point mass at mu. sd < 0 is an error and returns NaN.
	 * 
	 */
	public static double dnorm(double x, double mean, double sd) {
		jdistlib.Normal nd = new jdistlib.Normal(mean,sd);
		return nd.density(x,false);
	}

	public static double dnorm(double x, double mean, double sd, boolean log) {
		jdistlib.Normal nd = new jdistlib.Normal(mean,sd);
		return nd.density(x,log);
	}
	
	public static double pnorm(double x, boolean useLog) {
		jdistlib.Normal nd = new jdistlib.Normal(0.0,1.0);
		return nd.cumulative(x,true,useLog);
	}
	
	public static double pnorm(double x) {
		jdistlib.Normal nd = new jdistlib.Normal(0.0,1.0);
		return nd.cumulative(x,true,false);
	}
	
	public static double pnorm(double x, double mean, double sd) {
		jdistlib.Normal nd = new jdistlib.Normal(mean,sd);
		return nd.cumulative(x,true,false);
	}

	public static double pnorm(double x, double mean, double sd, boolean lower_tail, boolean log_p) {
		jdistlib.Normal nd = new jdistlib.Normal(mean,sd);
		return nd.cumulative(x,lower_tail,log_p);
	}

	public static double rnorm() {
		jdistlib.Normal nd = new jdistlib.Normal(0.0,1.0);
		return nd.random();
	}
	
	public static double rnorm(double mean, double sd) {
		jdistlib.Normal nd = new jdistlib.Normal(mean,sd);
		return nd.random();
	}
	
	public static double qnorm(double x, double mean, double sd) {
		jdistlib.Normal nd = new jdistlib.Normal(mean,sd);
		return nd.quantile(x);
	}
	
	public static double qnorm(double x, double mean, double sd, boolean lower_tail, boolean log_p) {
		jdistlib.Normal nd = new jdistlib.Normal(mean,sd);
		return nd.quantile(x,lower_tail,log_p);
	}

	/**
	 * The Exponential Distribution
	 * 
	 * Description
	 * 
	 * Density, distribution function, quantile function and random 
	 * generation for the exponential distribution with rate rate 
	 * (i.e., mean 1/rate).
	 * 
	 * Usage
	 * 
	 * dexp(x, rate = 1, log = FALSE)
	 * pexp(q, rate = 1, lower.tail = TRUE, log.p = FALSE)
	 * qexp(p, rate = 1, lower.tail = TRUE, log.p = FALSE)
	 * rexp(n, rate = 1)
	 * 
	 * Arguments
	 * x, q	
	 * 	vector of quantiles.
	 * p	
	 * 	vector of probabilities.
	 * n	
	 * 	number of observations. If length(n) > 1, the length is taken to be the number required.
	 * rate	
	 * 	vector of rates.
	 * log, log.p	
	 * 	logical; if TRUE, probabilities p are given as log(p).
	 * lower.tail	
	 * 	logical; if TRUE (default), probabilities are P[X ≤ x], otherwise, P[X > x].
	 * 
	 * @param bound
	 * @param rate
	 * @param ltail
	 * @param logflag
	 * @return
	 */
	public static double dexp(double bound, double rate, int ltail, boolean logflag) {
		return jdistlib.Exponential.density(bound, 1/rate, logflag);
	}

	public static double pexp(double bound, double rate) {
		//ExponentialDistribution exp = new ExponentialDistribution(0.0);
		//return logflag ? exp.cumulativeProbability(bound) : log(exp.cumulativeProbability(bound));
		return pexp(bound, rate, true, false);
	}
	
	public static double pexp(double bound, double rate, boolean ltail, boolean logflag) {
		// RMath pexp is parametrisized by scale = 1 / rate
		return jdistlib.Exponential.cumulative(bound, 1/rate, ltail, logflag);
	}
	
	public static double qexp(double u2, double rate_lambda) {
		// RMath qexp is parametrisized by scale = 1 / rate
		return jdistlib.Exponential.quantile(u2, 1 / rate_lambda, true, false);
	}

	public static double qexp(double u2, double rate_lambda, boolean ltail, boolean logflag) {
		// RMath qexp is parametrisized by scale = 1 / rate
		return jdistlib.Exponential.quantile(u2, 1 / rate_lambda, ltail, logflag);
	}

	public static double rexp(double rate) {
		return jdistlib.Exponential.random(1/rate, mt);
	}

	/**
	 * Draw from right truncated Exponential Distribution
	 * 
	 * @param rate (1/scale)
	 * @param highBound right truncation point
	 * @return right truncated exponential covariate
	 */
	public static double rtexp(double rate, double highBound) {
		return TruncatedExponential.rtexp(rate, highBound);
	}

	/**
	 * The Student t Distribution
	 * 
	 * Description
	 * 
	 * Density, distribution function, quantile function and random generation 
	 * for the t distribution with df degrees of freedom (and optional 
	 * non-centrality parameter ncp).
	 * 
	 * Usage
	 * 
	 * dt(x, df, ncp, log = FALSE)
	 * pt(q, df, ncp, lower.tail = TRUE, log.p = FALSE)
	 * qt(p, df, ncp, lower.tail = TRUE, log.p = FALSE)
	 * rt(n, df, ncp)
	 * 
	 * Arguments
	 * x, q	
	 * 	vector of quantiles.
	 * p	
	 * 	vector of probabilities.
	 * n	
	 * 	number of observations. If length(n) > 1, the length is taken to be the number required.
	 * df	
	 * 	degrees of freedom (> 0, maybe non-integer). df = Inf is allowed.
	 * ncp	
	 * 	non-centrality parameter delta; currently except for rt(), only for abs(ncp) <= 37.62. If omitted, use the central t distribution.
	 * log, log.p	
	 * 	logical; if TRUE, probabilities p are given as log(p).
	 * lower.tail	
	 * 	logical; if TRUE (default), probabilities are P[X ≤ x], otherwise, P[X > x].
	 * 
	 * @param x
	 * @param mean
	 * @param degrees
	 * @return
	 */
	public static double dt(double x, double degrees) {
		return jdistlib.T.density(x,degrees,false);
	}

	public static double rt(double degrees) {
		return jdistlib.T.random(degrees, mt);
	}

	public static double [] rt(int samples, double degrees) {
		return jdistlib.T.random(samples, degrees, mt);	
	}


	/**
	 * 
	 * The Gamma Distribution
	 * 
	 * Description
	 * 
	 * Density, distribution function, quantile function and random generation for the 
	 * Gamma distribution with parameters shape and scale.
	 * 
	 * Usage
	 * 
	 * dgamma(x, shape, rate = 1, scale = 1/rate, log = FALSE)
	 * pgamma(q, shape, rate = 1, scale = 1/rate, lower.tail = TRUE,
	 *        log.p = FALSE)
	 * qgamma(p, shape, rate = 1, scale = 1/rate, lower.tail = TRUE,
	 *        log.p = FALSE)
	 * rgamma(n, shape, rate = 1, scale = 1/rate)
	 * Arguments
	 * 
	 * x, q	
	 * vector of quantiles.
	 * 
	 * p	
	 * vector of probabilities.
	 * 
	 * n	
	 * number of observations. If length(n) > 1, the length is taken to be the number 
	 * required.
	 * 
	 * rate	
	 * an alternative way to specify the scale.
	 * 
	 * shape, scale	
	 * shape and scale parameters. Must be positive, scale strictly.
	 * 
	 * log, log.p	
	 * logical; if TRUE, probabilities/densities p are returned as log(p).
	 * 
	 * lower.tail	
	 * logical; if TRUE (default), probabilities are P[X ≤ x], otherwise, P[X > x].
	 * 
	 * Details
	 * 
	 * If scale is omitted, it assumes the default value of 1.
	 * 
	 * The Gamma distribution with parameters shape = a and scale = s has density
	 * 
	 * f(x)= 1/(s^a Gamma(a)) x^(a-1) e^-(x/s)
	 * 
	 * for x ≥ 0, a > 0 and s > 0. (Here Gamma(a) is the function implemented by R's 
	 * gamma() and defined in its help. Note that a = 0 corresponds to the trivial 
	 * distribution with all mass at point 0.)
	 * 
	 * The mean and variance are E(X) = a*s and Var(X) = a*s^2.
	 * 
	 * The cumulative hazard H(t) = - log(1 - F(t)) is
	 * 
	 * -pgamma(t, ..., lower = FALSE, log = TRUE)
	 * Note that for smallish values of shape (and moderate scale) a large parts of 
	 * the mass of the Gamma distribution is on values of x so near zero that they 
	 * will be represented as zero in computer arithmetic. So rgamma may well return 
	 * values which will be represented as zero. (This will also happen for very large 
	 * values of scale since the actual generation is done for scale = 1.)
	 * Value
	 * 
	 * dgamma gives the density, pgamma gives the distribution function, qgamma 
	 * gives the quantile function, and rgamma generates random deviates.
	 * 
	 * Invalid arguments will result in return value NaN, with a warning.
	 * 
	 * The length of the result is determined by n for rgamma, and is the maximum of 
	 * the lengths of the numerical arguments for the other functions.
	 * 
	 * The numerical arguments other than n are recycled to the length of the result. 
	 * Only the first elements of the logical arguments are used.
	 * 
	 * Note
	 * 
	 * The S (Becker et al (1988) parametrization was via shape and rate: S had no 
	 * scale parameter. In R 2.x.y scale took precedence over rate, but now it is an error to supply both.
	 * 
	 * pgamma is closely related to the incomplete gamma function. As defined by
	 *  Abramowitz and Stegun 6.5.1 (and by ‘Numerical Recipes’) this is
	 * 
	 * P(a,x) = 1/Gamma(a) integral_0^x t^(a-1) exp(-t) dt
	 * 
	 * P(a, x)
	 * is pgamma(x, a). Other authors (for example Karl Pearson in his 1922 tables) 
	 * omit the normalizing factor, defining the incomplete gamma function γ(a,x) as 
	 * gamma(a,x) = integral_0^x t^(a-1) exp(-t) dt, i.e., pgamma(x, a) * gamma(a). 
	 * Yet other use the ‘upper’ incomplete gamma function,
	 * 
	 * Gamma(a,x) = integral_x^Inf t^(a-1) exp(-t) dt,
	 * 
	 * which can be computed by pgamma(x, a, lower = FALSE) * gamma(a).
	 * 
	 * Note however that pgamma(x, a, ..) currently requires a > 0, whereas the 
	 * incomplete gamma function is also defined for negative a. In that case, you 
	 * can use gamma_inc(a,x) (for Γ(a,x)) from package gsl.
	 * 
	 * See also http://en.wikipedia.org/wiki/Incomplete_gamma_function, or
	 *  http://dlmf.nist.gov/8.2#i.
	 * 
	 * Source
	 * 
	 * dgamma is computed via the Poisson density, using code contributed by 
	 * Catherine Loader (see dbinom).
	 * 
	 * pgamma uses an unpublished (and not otherwise documented) algorithm 
	 * ‘mainly by Morten Welinder’.
	 * 
	 * qgamma is based on a C translation of
	 * 
	 * Best, D. J. and D. E. Roberts (1975). Algorithm AS91. Percentage points of 
	 * the chi-squared distribution. Applied Statistics, 24, 385–388.
	 * 
	 * plus a final Newton step to improve the approximation.
	 * 
	 * rgamma for shape >= 1 uses
	 * 
	 * Ahrens, J. H. and Dieter, U. (1982). Generating gamma variates by a modified
	 *  rejection technique. Communications of the ACM, 25, 47–54,
	 * 
	 * and for 0 < shape < 1 uses
	 * 
	 * Ahrens, J. H. and Dieter, U. (1974). Computer methods for sampling from gamma, 
	 * beta, Poisson and binomial distributions. Computing, 12, 223–246.
	 * 
	 * References
	 * 
	 * Becker, R. A., Chambers, J. M. and Wilks, A. R. (1988) The New S Language. 
	 * Wadsworth & Brooks/Cole.
	 * 
	 * Shea, B. L. (1988) Algorithm AS 239, Chi-squared and incomplete Gamma integral,
	 *  Applied Statistics (JRSS C) 37, 466–473.
	 * 
	 * Abramowitz, M. and Stegun, I. A. (1972) Handbook of Mathematical Functions. 
	 * New York: Dover. Chapter 6: Gamma and Related Functions.
	 * 
	 * NIST Digital Library of Mathematical Functions. 
	 * http://dlmf.nist.gov/, section 8.2.
	 * 
	 * See Also
	 * 
	 * gamma for the gamma function.
	 * 
	 * Distributions for other standard distributions, including dbeta for the 
	 * Beta distribution and dchisq for the chi-squared distribution which is a 
	 * special case of the Gamma distribution.
	 * 

	 * 
	 * @param u
	 * @param shape
	 * @param scale
	 * @param ltail
	 * @param logflag
	 * @return
	 */
	public static double dgamma(double u, double shape, double scale) {
		return jdistlib.Gamma.density(u,shape,scale,false);
	}

	public static double dgamma(double u, double shape, double scale, boolean logflag) {
		return jdistlib.Gamma.density(u,shape,scale,logflag);
	}

	public static double pgamma(double x, double shape, double scale) {
		return jdistlib.Gamma.cumulative(x, shape, scale, true, false);
	}

	public static double pgamma(double x, double shape, double scale, boolean lower_tail, boolean log_p) {
		return jdistlib.Gamma.cumulative(x, shape, scale, lower_tail, log_p);	
	}

	public static double qgamma(double x, double shape, double scale, boolean lower_tail, boolean log_p) {
		return jdistlib.Gamma.quantile(x, shape, scale, lower_tail, log_p);
	}
	
	public static double qgamma(double x, double shape, double scale) {	    
		return jdistlib.Gamma.quantile(x, shape, scale, true, false);
	}

	public static double rgamma(double a, double scale) {
		return jdistlib.Gamma.random(a, scale, mt);
	}

	/**
	 * Beta related methods
	 * 
	 * @param a shape
	 * @param b shape
	 * 
	 * @return beta distributed covariate
	 */
	/*public static double rbeta(double a, double b) {
		double Ga = Gamma.rgamma(a);
		double Gb = Gamma.rgamma(b);
		return Ga / (Ga + Gb);
	}*/
}
