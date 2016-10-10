package xyz.lejon.bayes.models.dolda;

import static org.ejml.ops.CommonOps.multInner;
import static org.ejml.ops.CommonOps.multTransA;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

import org.apache.commons.configuration.ConfigurationException;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import xyz.lejon.configuration.DOLDAConfiguration;
import xyz.lejon.sampling.FastMVNSamplerCholesky;
import xyz.lejon.sampling.InverseGammaDistribution;
import xyz.lejon.utils.EJMLUtils;

/**
 * This class implements the samples the beta coefficients using 
 * ordinary Bayesian Linear Regression (BLR) including sampling 
 * sigma squared
 * 
 * @author Leif Jonsson
 *
 */
public class DOLDAGibbsEJMLNormal extends EDOLDANormal {

	private static final long serialVersionUID = 1L;

	private ForkJoinPool betaSamplerPool;

	public DOLDAGibbsEJMLNormal(DOLDAConfiguration parentCfg, double[][] xs, int[] iys) throws ConfigurationException {
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
	
	public double sampleSigmaSquared(DenseMatrix64F mu, DenseMatrix64F Lambda, DenseMatrix64F yty) {
		double vn = v0  + ys.length / (double) 2;
		DenseMatrix64F tmp = null;
		synchronized (betaLock) {
			DenseMatrix64F t1 = new DenseMatrix64F(Lambda.numRows, mu.numCols);
			CommonOps.mult(Lambda, mu, t1);
			DenseMatrix64F t2 = new DenseMatrix64F(mu.numCols, t1.numCols);
			CommonOps.multTransA(mu, t1, t2);

			// yty.sub(mu.transpose().mmul(Lambda.mmul(mu)));
			tmp = new DenseMatrix64F(yty.numRows, t2.numCols);
			CommonOps.subtract(yty, t2, tmp);
		}
		
		if(tmp.numRows!=1 && tmp.numCols!=1) {
			throw new IllegalStateException("intermediary has wrong dims!");
		}
		
		double taun = tau0 + tmp.get(0) / 2;
		
		InverseGammaDistribution ig = new InverseGammaDistribution(vn, taun);

		return ig.nextInverseGamma();
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
			// After this operation the precision is inverted!
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
		DenseMatrix64F yty = new DenseMatrix64F(zColKd.numCols,zColKd.numCols);
		synchronized (betaLock) {
			multInner(zColKd, yty);
		}
		SigmaSq[k] = sampleSigmaSquared(localMu, precision, yty);
	}
}
