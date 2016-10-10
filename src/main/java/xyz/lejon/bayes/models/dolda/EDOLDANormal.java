package xyz.lejon.bayes.models.dolda;

import org.apache.commons.configuration.ConfigurationException;

import xyz.lejon.configuration.DOLDAConfiguration;

public abstract class EDOLDANormal extends EDOLDA {

	private static final long serialVersionUID = 1L;
	
	public static final double c = 100;
	protected double [] SigmaSq;
	protected final double v0 = 1.0;
	protected final double tau0 = 1.0;

	public EDOLDANormal(DOLDAConfiguration parentCfg, double[][] xs, int[] ys) throws ConfigurationException {
		super(parentCfg, xs, ys);
		SigmaSq = new double[noClasses];
	}
}
