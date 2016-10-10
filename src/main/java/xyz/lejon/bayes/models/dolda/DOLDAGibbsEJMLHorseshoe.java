package xyz.lejon.bayes.models.dolda;

import static org.ejml.ops.CommonOps.multTransA;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

import org.apache.commons.configuration.ConfigurationException;
import org.ejml.data.DenseMatrix64F;

import xyz.lejon.configuration.DOLDAConfiguration;
import xyz.lejon.sampling.FastMVNSamplerCholesky;
import xyz.lejon.utils.EJMLUtils;
import xyz.lejon.utils.MatrixOps;

/**
 * This class implements the DO_Probit horseshoe sampler mentioned in 
 * 'Diagonal Orthant Multinomial Probit Models', James E. Johndrow, Kristian Lum and David B. Dunson
 * 
 * @author Leif Jonsson
 *
 */
public class DOLDAGibbsEJMLHorseshoe extends EDOLDAHorseshoe {
	
	private static final long serialVersionUID = 1L;
	private ForkJoinPool betaSamplerPool;

	public DOLDAGibbsEJMLHorseshoe(DOLDAConfiguration parentCfg, double[][] xs, int[] ys) throws ConfigurationException {
		super(parentCfg, xs, ys);
		//betaSamplerPool = new ForkJoinPool();
		betaSamplerPool = new ForkJoinPool(Runtime.getRuntime().availableProcessors(), ForkJoinPool.defaultForkJoinWorkerThreadFactory, parentCfg.getUncaughtExceptionHandler(), true);
		System.out.println("Using Beta sampler: " + this.getClass().getName());
	}
	
	class BetaSampler extends RecursiveAction {
		final static long serialVersionUID = 1L;
		int startClass = -1;
		int endClass = -1;
		int limit = 1;

		double [][] Xs;
		double [][] XtXs;
		double [][] Xts;
		double [][] precisions;
		

		public BetaSampler(int startClass, int endClass, double [][]  Xs, double [][] XtXs, double [][] Xts, double [][] precisions, int ll) {
			this.Xs = Xs;
			this.XtXs = XtXs;
			this.Xts = Xts;
			this.precisions = precisions;

			this.limit = ll;
			this.startClass = startClass;
			this.endClass = endClass;
		}

		@Override
		protected void compute() {
			if ( (endClass-startClass) <= limit ) {
				for (int classNo = startClass; classNo < endClass; classNo++) {
					sampleBeta(Xs, XtXs, Xts, MatrixOps.clone(precisions), classNo);
				}
			}
			else {
				int range = (endClass-startClass);
				int startClass1 = startClass;
				int endClass1 = startClass + (range / 2);
				int startClass2 = endClass1;
				int endClass2 = endClass;
				invokeAll(new BetaSampler(startClass1,endClass1, Xs, XtXs, Xts, precisions, limit),
						new BetaSampler(startClass2,endClass2, Xs, XtXs, Xts, precisions, limit));
			}
		}
	}

	@Override
	protected void sampleBetas(double [][] Xs, double [][] XtXs, double [][] Xts, double [][] precisions) {
		BetaSampler bs = new BetaSampler(0, noClasses,Xs, XtXs, Xts, precisions,10);                
		betaSamplerPool.invoke(bs); 
	}
	
	void sampleBeta(double [][] Xs, double [][] XtXs, double [][] Xts, double [][] precisions,  int k) {	
		// Code review done Mans and Leif: 2015-11-02
		
		DenseMatrix64F X  = new DenseMatrix64F(Xs);
		DenseMatrix64F precision = new DenseMatrix64F(precisions);
		
		// Update diagonal elements of precision matrix
		for (int i = 0; i < precision.numRows; i++) {
			double update;
			// The intercept should not be regularized with the horseshoe
			if(i==0 && useIntecept) { 
				update = EDOLDANormal.c;
			} else {
				update = (Math.pow(Tau[k], 2) * Math.pow(Lambda[k][i],2));
			}
			precision.set(i, i, XtXs[i][i] + 1 / update);
		}

		double [] zColk = Ast[k];
		DenseMatrix64F zColKd = new DenseMatrix64F(zColk.length,1);
		zColKd.setData(zColk);
		DenseMatrix64F localTilde = new DenseMatrix64F(X.numCols, 1);
		DenseMatrix64F localMu = new DenseMatrix64F(X.numCols, 1);
		
		// It SEEMS that multTransA and solve are thread safe... Yes I know, shrug... :(
		//synchronized (betaLock) {
			multTransA(X, zColKd, localTilde);

			// After this operation precision is inverted!
			//invert(precision);
			//CommonOps.mult(precision, localTilde, localMu);
			//FastMVNSampler mvns = new FastMVNSampler(localMu, precision);

			EJMLUtils.solveCovariace(precision,localTilde,localMu);
		//}
		
		//FastMVNSamplerEJML mvns = new FastMVNSamplerEJML(localMu, precision, true);
		FastMVNSamplerCholesky mvns = new FastMVNSamplerCholesky(localMu, precision, true);
		betas[k] = mvns.sample();
		
		sampleTauAndLambda(k);
	}

}
