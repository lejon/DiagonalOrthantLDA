package xyz.lejon.bayes.models.dolda;

import org.apache.commons.configuration.ConfigurationException;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import xyz.lejon.configuration.DOLDAConfiguration;
import xyz.lejon.sampling.FastMVNSamplerJBLAS;

/**
 * This class implements the DO_Probit normal sampler as described in 
 * 'Diagonal Orthant Multinomial Probit Models', James E. Johndrow, Kristian Lum and David B. Dunson
 * 
 * It differs from plain BLR in that it does not sample sigma squared.
 * 
 * @author Leif Jonsson
 *
 */
public class DOLDAGibbsJBlasDOProbitNormal extends EDOLDANormal {

	private static final long serialVersionUID = 1L;

	public DOLDAGibbsJBlasDOProbitNormal(DOLDAConfiguration parentCfg, double[][] xs, int[] iys) throws ConfigurationException {
		super(parentCfg, xs, iys);
		System.out.println("Using Beta sampler: " + this.getClass().getName());
	}
	
	@Override
	protected void sampleBetas(double [][] Xs, double [][] XtXs, double [][] Xts, double [][] precisions) {
		
		// Update diagonal elements of precision matrix
		for (int i = 0; i < precisions.length; i++) {
			precisions[i][i] =  XtXs[i][i] + (1 / c);
		}
		
		for (int k = 0; k < noClasses; k++) {
			sampleBeta(Xs, XtXs, Xts, precisions, k);
			saveBetaSample(k);
		}
	}

	protected void sampleBeta(double [][] Xs, double [][] XtXs, double [][] Xts, double [][] precisions,  int k) {
		// Code review done Mans and Leif: 2015-11-02

		DoubleMatrix X = new DoubleMatrix(Xs);
		DoubleMatrix Xt = new DoubleMatrix(Xts);
		DoubleMatrix precision = new DoubleMatrix(precisions);

		double [] zColk = Ast[k]; // Our latent variables, a^T (transposed colvector for category k). 
		DoubleMatrix zColKd = new DoubleMatrix(zColk);
		//DoubleMatrix pInv = null;
		DoubleMatrix localMu = new DoubleMatrix(X.columns, 1);
		synchronized (betaLock) {
			DoubleMatrix localTilde = Xt.mmul(zColKd); // X^T * a_k 
			localMu = Solve.solve(precision, localTilde); // precision^-1 * localTilde => (X^T * X + L)^-1 X^T a_k
			//localMu = pInv.mmul(localTilde);
			//pInv = BlasOps.blasInvert(precision);
		}
		
		//FastMVNSamplerJBLAS mvn = new FastMVNSamplerJBLAS(pInv);
		FastMVNSamplerJBLAS mvn = new FastMVNSamplerJBLAS(precision, true);
		betas[k] = mvn.sample(localMu);
	}
}
