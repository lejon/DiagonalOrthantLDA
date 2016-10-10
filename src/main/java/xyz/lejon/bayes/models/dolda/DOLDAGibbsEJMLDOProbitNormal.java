package xyz.lejon.bayes.models.dolda;

import static org.ejml.ops.CommonOps.multTransA;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

import org.apache.commons.configuration.ConfigurationException;
import org.ejml.data.DenseMatrix64F;

import xyz.lejon.configuration.DOLDAConfiguration;
import xyz.lejon.sampling.FastMVNSamplerCholesky;
import xyz.lejon.utils.EJMLUtils;

/**
 * This class implements the DO_Probit normal sampler as described in 
 * 'Diagonal Orthant Multinomial Probit Models', James E. Johndrow, Kristian Lum and David B. Dunson
 * 
 * It differs from plain BLR in that it does not sample sigma squared.
 * 
 * @author Leif Jonsson
 *
 */
public class DOLDAGibbsEJMLDOProbitNormal extends EDOLDANormal {

	private static final long serialVersionUID = 1L;

	private ForkJoinPool betaSamplerPool;

	public DOLDAGibbsEJMLDOProbitNormal(DOLDAConfiguration parentCfg, double[][] xs, int[] iys) throws ConfigurationException {
		super(parentCfg, xs, iys);
		betaSamplerPool = new ForkJoinPool();
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
			try {
				if ( (endClass-startClass) <= limit ) {
					for (int classNo = startClass; classNo < endClass; classNo++) {
						sampleBeta(Xs, XtXs, Xts, precisions, classNo);
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
			catch ( Exception e ) {
				e.printStackTrace();
			}
		}
	}

	@Override
	protected void sampleBetas(double [][] Xs, double [][] XtXs, double [][] Xts, double [][] precisions) {
		// Update diagonal elements of precision matrix
		for (int i = 0; i < precisions.length; i++) {
			precisions[i][i] =  XtXs[i][i] + (1 / c);
		}
		BetaSampler bs = new BetaSampler(0, noClasses, Xs, XtXs, Xts, precisions, 10);                
		betaSamplerPool.invoke(bs); 
	}
	
	private void sampleBeta(double [][] Xs, double [][] XtXs, double [][] Xts, double [][] precisions,  int k) {		
		DenseMatrix64F X  = new DenseMatrix64F(Xs);
		DenseMatrix64F precision = new DenseMatrix64F(precisions);

		double [] zColk = Ast[k];
		DenseMatrix64F zColKd = new DenseMatrix64F(zColk.length,1);
		zColKd.setData(zColk);
		DenseMatrix64F localTilde = new DenseMatrix64F(X.numCols, 1);
		DenseMatrix64F localMu = new DenseMatrix64F(X.numCols, 1);

		// It SEEMS that multTransA and solve are thread safe... Yes I know, shrug... :(
		//synchronized (betaLock) {
			multTransA(X, zColKd, localTilde);

			// After this operation precision is inverted!
			// IF THIS SOLUTION IS reintroduced it MUST be ensured that precision 
			// is a local copy since it is shared between threads!
			//invert(precision);
			//CommonOps.mult(precision, localTilde, localMu);
			//FastMVNSampler mvns = new FastMVNSampler(localMu, precision);

			EJMLUtils.solveCovariace(precision,localTilde,localMu);
		//}
		//FastMVNSamplerEJML mvns = new FastMVNSamplerEJML(localMu, precision, true);
		FastMVNSamplerCholesky mvns = new FastMVNSamplerCholesky(localMu, precision, true);
		betas[k] = mvns.sample();
	}
}
