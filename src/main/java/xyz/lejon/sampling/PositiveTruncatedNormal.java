package xyz.lejon.sampling;

public class PositiveTruncatedNormal extends TruncatedNormal {

	double low = 0.0;
	double high = Double.POSITIVE_INFINITY;
		
	public double rand( double mean, double sd ) {
		return rand(mean,sd, low, high);
	}
	
}
