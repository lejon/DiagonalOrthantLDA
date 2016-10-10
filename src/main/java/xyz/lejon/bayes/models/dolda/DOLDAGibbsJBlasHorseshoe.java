package xyz.lejon.bayes.models.dolda;

import org.apache.commons.configuration.ConfigurationException;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import xyz.lejon.configuration.DOLDAConfiguration;
import xyz.lejon.sampling.FastMVNSamplerJBLAS;
import xyz.lejon.utils.BlasOps;

/**
 * This class implements the DO_Probit horseshoe sampler mentioned in 
 * 'Diagonal Orthant Multinomial Probit Models', James E. Johndrow, Kristian Lum and David B. Dunson
 * 
 * @author Leif Jonsson
 *
 */
public class DOLDAGibbsJBlasHorseshoe extends EDOLDAHorseshoe {

	private static final long serialVersionUID = 1L;

	public DOLDAGibbsJBlasHorseshoe(DOLDAConfiguration parentCfg, double[][] xs, int[] ys) throws ConfigurationException {
		super(parentCfg, xs, ys);
		System.out.println("Using Beta sampler: " + this.getClass().getName());
	}
	
	@Override
	protected void sampleBetas(double [][] Xs, double [][] XtXs, double [][] Xts, double [][] precisions) {
		for (int k = 0; k < noClasses; k++) {
			sampleBeta(Xs, XtXs, Xts, precisions, k);
			saveBetaSample(k);
		}
	}
	
	protected void sampleBeta(double [][] Xs, double [][] XtXs, double [][] Xts, double [][] precisions,  int k) {		
		DoubleMatrix X = new DoubleMatrix(Xs);
		DoubleMatrix Xt = new DoubleMatrix(Xts);
		DoubleMatrix precision = new DoubleMatrix(precisions);
	
		// Update diagonal elements of precision matrix
		for (int i = 0; i < precision.rows; i++) {
			double update;
			// The intercept should not be regularized with the horseshoe
			if(i==0 && useIntecept) { 
				update =  EDOLDANormal.c;
			} else {
				update = (Math.pow(Tau[k], 2) * Math.pow(Lambda[k][i],2));
			}
			precision.put(i, i, XtXs[i][i] + 1 / update);
		}

		double [] zColk = Ast[k];
		DoubleMatrix zColKd = new DoubleMatrix(zColk);
		DoubleMatrix localTilde = null;
		DoubleMatrix localMu = null;
		DoubleMatrix pInv = null;
		synchronized (betaLock) {
			localTilde = Xt.mmul(zColKd);
			localMu = new DoubleMatrix(X.columns, 1);

			pInv = null;
			localMu = Solve.solve(precision, localTilde);
			pInv = BlasOps.blasInvert(precision);
			//localMu = pInv.mmul(localTilde);
		}

		// We don't have to multiply with Sigma here since it is always 1
		FastMVNSamplerJBLAS mvn = new FastMVNSamplerJBLAS(pInv);
		//FastMVNSamplerJBLAS mvn = new FastMVNSamplerJBLAS(precision, true);
		betas[k] = mvn.sample(localMu);
		
		sampleTauAndLambda(k);
	}

}
