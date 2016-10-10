package xyz.lejon.bayes.models.dolda;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.configuration.ConfigurationException;

import xyz.lejon.bayes.models.probit.HorseshoeDOProbit;
import xyz.lejon.configuration.DOLDAConfiguration;

public abstract class EDOLDAHorseshoe extends EDOLDA {

	private static final long serialVersionUID = 1L;
	List<double []> [] sampledLambdas;
	List<Double> [] sampledTaus;
	double [] Tau;
	double [][] Lambda;

	@SuppressWarnings("unchecked")
	public EDOLDAHorseshoe(DOLDAConfiguration parentCfg, double[][] xs, int[] ys) throws ConfigurationException {
		super(parentCfg, xs, ys);
		Tau = new double [noClasses];
		Arrays.fill(Tau, 1.0); 
		Lambda = new double [noClasses][p+ks];
		for (int i = 0; i < Lambda.length; i++) {
			Arrays.fill(Lambda[i], 1.0); 
		}
		sampledLambdas = (ArrayList<double []>[]) new ArrayList<?>[noClasses];
		sampledTaus    = (ArrayList<Double>[]) new ArrayList<?>[noClasses];
		for (int l = 0; l < noClasses; l++) {
			sampledLambdas[l] = new ArrayList<double []>();
			sampledTaus[l]    = new ArrayList<Double>();
		}
	}
	
	 void sampleTauAndLambda(int k) {
		Lambda[k] = HorseshoeDOProbit.sampleLambda(Tau[k], betas[k], Lambda[k], Sigma);
		Tau[k]    = HorseshoeDOProbit.sampleTau(Tau[k], betas[k], Lambda[k], Sigma, useIntecept);
		sampledTaus[k].add(Tau[k]);
		sampledLambdas[k].add(Lambda[k].clone());
	}
}
