package xyz.lejon.sampling;

public class NegativeTruncatedNormal extends TruncatedNormal {

	double low = Double.NEGATIVE_INFINITY;
	double high = 0.0;
		
	public double rand( double mean, double sd ) {
		return rand(mean,sd, low, high);
	}
	
}
