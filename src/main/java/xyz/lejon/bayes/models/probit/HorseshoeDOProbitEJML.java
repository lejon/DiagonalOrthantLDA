package xyz.lejon.bayes.models.probit;

import static org.ejml.ops.CommonOps.invert;
import static org.ejml.ops.CommonOps.multTransA;
import static xyz.lejon.utils.MatrixOps.rnorm;

import java.io.IOException;
import java.util.Arrays;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import xyz.lejon.configuration.DOConfiguration;
import xyz.lejon.sampling.FastMVNSamplerEJML;
import xyz.lejon.utils.MatrixOps;

//public class HorseshoeDOProbit extends MultivariateParallelDOSampler {
public class HorseshoeDOProbitEJML extends HorseshoeDOProbit {

	MatrixOps mo = new MatrixOps();
	double [] Tau;
	double [] Sigma;
	double [][] Lambda;

	public HorseshoeDOProbitEJML(DOConfiguration config, double[][] xs, int[] ys, int noClasses) throws IOException {
		super(config, xs, ys, noClasses);
		Lambda = new double[noClasses][];
		for (int i = 0; i < Lambda.length; i++) {
			Lambda[i] = new double[noCovariates];
			Arrays.fill(Lambda[i], 1.0);
		}
		Tau = new double[noClasses];
		Arrays.fill(Tau, 1.0);
		Sigma = new double[noClasses];
		Arrays.fill(Sigma, 1.0);
		for (int i = 0; i < noClasses; i++) {
			for (int j = 0; j < noCovariates; j++) {
				betas[i][j] = rnorm(0.0, 1.0);
			}
		}
		DenseMatrix64F Xdtrans = Xd.copy();
		CommonOps.transpose(Xdtrans);
		//printBeta = true;
		iterinter = 100;
		System.out.println("################# Running HORSESHOE EJML #################");
	}

	@Override
	public double [] sampleBeta(int k) {
		double [] zColk = zsT[k];
		
		Tau[k]    = sampleTau(Tau[k], betas[k], Lambda[k], Sigma[k], useIntecept);
		Lambda[k] = sampleLambda(Tau[k], betas[k], Lambda[k], Sigma[k]);

		DenseMatrix64F myPrecision = new DenseMatrix64F(noCovariates,noCovariates);
		// Rebuild updated precision matrix
		for (int i = 0; i < myPrecision.numRows; i++) {
			for (int j = 0; j < myPrecision.numCols; j++) {
				if(i==j) {
					double update;
					if(i==0 && useIntecept) {
						update = c;
					} else {
						update = Math.pow(Tau[k], 2) * Math.pow(Lambda[k][i],2);
					}
					myPrecision.set(i, j, XtX.get(i,j)  + update);
				} else {
					myPrecision.set(i, j, XtX.get(i,j));
				}
			}
		}

		invert(myPrecision);

		DenseMatrix64F zColKd = new DenseMatrix64F(zColk.length,1);
		zColKd.setData(zColk);
		DenseMatrix64F localTilde = new DenseMatrix64F(muTilde);
		DenseMatrix64F localMu = new DenseMatrix64F(Xd.numCols, 1);
		multTransA(Xd, zColKd, localTilde);
		CommonOps.mult(myPrecision, localTilde, localMu);

		FastMVNSamplerEJML mvns = new FastMVNSamplerEJML(localMu, myPrecision);
		betas[k] = mvns.sample();
		return betas[k];
	}
}
