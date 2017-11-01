package xyz.lejon.bayes.models.probit;
import static org.ejml.ops.CommonOps.addEquals;
import static org.ejml.ops.CommonOps.invert;
import static org.ejml.ops.CommonOps.multInner;
import static org.ejml.ops.CommonOps.multTransA;

import java.io.IOException;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import xyz.lejon.configuration.DOConfiguration;
import xyz.lejon.sampling.FastMVNSamplerEJML;
import xyz.lejon.utils.MatrixOps;


public class MultivariateParallelDOSampler extends AbstractParallelDOSampler implements DOSampler {
	protected DenseMatrix64F Xd;
	protected DenseMatrix64F XtX;
	protected DenseMatrix64F muTilde;
	protected DenseMatrix64F mumu;
	protected double c = 10;
	FastMVNSamplerEJML mvns; 

	public MultivariateParallelDOSampler(DOConfiguration config, double [][] xs, int [] ys, int noClasses) throws IOException {
		this.xs = xs;
		this.ys = ys;
		this.noClasses = noClasses;
		setupSampler(config, xs, noClasses);
		double [] tmpMean = new double[noCovariates];
		mvns = new FastMVNSamplerEJML(tmpMean, Stilde);
	}
	
	protected void setupSampler(DOConfiguration config, double[][] xs, int noClasses) {

		super.setupSampler(config, xs, noClasses);

		Xd  = new DenseMatrix64F(xs);
		XtX = new DenseMatrix64F(Xd.numCols, Xd.numCols);
		mumu = new DenseMatrix64F(Xd.numCols, 1);
		muTilde = new DenseMatrix64F(Xd.numCols, 1);
		
		multInner(Xd, XtX);
		
		double [][] priorPrecMatrix = new double[Xd.numCols][Xd.numCols];
		for (int i = 0; i < priorPrecMatrix.length; i++) {
			for (int j = 0; j < priorPrecMatrix.length; j++) {
				if(i==j) {
					priorPrecMatrix[i][j] = 1 / c;
				}
			}
		}
		priorPrecision = new DenseMatrix64F(priorPrecMatrix);

		priorMean = new DenseMatrix64F(XtX);
		addEquals(priorMean, priorPrecision);
		invert(priorMean);
		Stilde = MatrixOps.extractDoubleArray(priorMean);
		double [] tmpMean = new double[noCovariates];
		mvns = new FastMVNSamplerEJML(tmpMean, Stilde);
	}

	public double [] sampleBeta(int k) {
		double [] zColk = zsT[k];

		DenseMatrix64F zColKd = new DenseMatrix64F(zColk.length,1);
		zColKd.setData(zColk);
		DenseMatrix64F localTilde = new DenseMatrix64F(muTilde);
		DenseMatrix64F localMu = new DenseMatrix64F(Xd.numCols, 1);
		multTransA(Xd, zColKd, localTilde);
		CommonOps.mult(priorMean, localTilde, localMu);
		double [] mu_tile = localMu.getData();

		betas[k] = mvns.sample(mu_tile);
		return betas[k];
	}
}
