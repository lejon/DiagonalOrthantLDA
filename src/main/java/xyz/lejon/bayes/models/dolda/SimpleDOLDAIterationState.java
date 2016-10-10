package xyz.lejon.bayes.models.dolda;

public class SimpleDOLDAIterationState implements DOLDAIterationState {

	double [][] phi;
	double [][] betas;
	
	public SimpleDOLDAIterationState(double[][] phi, double[][] betas) {
		super();
		this.phi = phi;
		this.betas = betas;
	}

	@Override
	public double[][] getPhi() {
		return phi;
	}

	@Override
	public double[][] getBetas() {
		return betas;
	}

}
