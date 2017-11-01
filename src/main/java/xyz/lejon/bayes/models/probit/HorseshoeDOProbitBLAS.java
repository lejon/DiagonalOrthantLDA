package xyz.lejon.bayes.models.probit;

import static xyz.lejon.utils.MatrixOps.rnorm;

import java.io.IOException;
import java.util.Arrays;

import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import xyz.lejon.configuration.DOConfiguration;
import xyz.lejon.sampling.FastMVNSamplerJBLAS;
import xyz.lejon.utils.BlasOps;
import xyz.lejon.utils.MatrixOps;
//public class HorseshoeDOProbit extends MultivariateParallelDOSampler {
public class HorseshoeDOProbitBLAS extends HorseshoeDOProbit {
	DoubleMatrix myXtX;
	DoubleMatrix myMuTilde;
	DoubleMatrix myXdtrans;
	DoubleMatrix myPrecision;

	public HorseshoeDOProbitBLAS(DOConfiguration config, double[][] xs, int[] ys, int noClasses) throws IOException {
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
		myXtX = new DoubleMatrix(MatrixOps.extractDoubleArray(XtX));
		myMuTilde = new DoubleMatrix(MatrixOps.extractDoubleArray(muTilde));
		myXdtrans = new DoubleMatrix(MatrixOps.extractDoubleArray(Xd.copy()));
		myXdtrans = myXdtrans.transpose();

		myPrecision = new DoubleMatrix(noCovariates,noCovariates);
		// Build precision matrix
		for (int i = 0; i < myPrecision.rows; i++) {
			for (int j = 0; j < myPrecision.columns; j++) {
				myPrecision.put(i, j,  myXtX.get(i, j));
			}
		}

		System.out.println("################# Running HORSESHOE BLAS #################");
	}

	@Override
	public double [] sampleBeta(int k) {
		// Rebuild updated precision matrix
		for (int i = 0; i < myPrecision.rows; i++) {
			double update;
			if(i==0 && useIntecept) {
				update = c;
			} else {
				update = (Math.pow(Tau[k], 2) * Math.pow(Lambda[k][i],2));
			}
			myPrecision.put(i, i, myXtX.get(i, i) + 1 / update);
		}

		double [] zColk = zsT[k];
		DoubleMatrix zColKd = new DoubleMatrix(zColk);
		DoubleMatrix localTilde = myXdtrans.mmul(zColKd);
		DoubleMatrix localMu = new DoubleMatrix(Xd.numCols, 1);

		localMu = Solve.solve(myPrecision, localTilde);
		DoubleMatrix pInv = BlasOps.blasInvert(myPrecision);
		//localMu = pInv.mmul(localTilde);

		betas[k]  = sampleBeta(localMu, pInv, Sigma[k]);
		for (int i = 0; i < betas[k].length; i++) {
			if(Double.isNaN(betas[k][i])) {
				throw new IllegalStateException("Betas contains NaN's:\n" 
					+ "mu: " + MatrixOps.arrToStr(localMu.toArray()) + "\n"
					+ "pInv:" + MatrixOps.doubleArrayToPrintString(pInv.toArray2(), 5, 5, 10));
			}
		}
		Lambda[k] = sampleLambda(Tau[k], betas[k], Lambda[k], Sigma[k]);
		Tau[k]    = sampleTau(Tau[k], betas[k], Lambda[k], Sigma[k], useIntecept);
		return betas[k];
	}
	
	public static double [] sampleBeta(DoubleMatrix mu, DoubleMatrix Sigma, double sigma) {
		FastMVNSamplerJBLAS mvn = new FastMVNSamplerJBLAS(Sigma.muli(sigma));
		return mvn.sample(mu);
	}
	
	protected void preIteration(int iter) {
		
	}

	protected void postIteration(int iter) {
		/*System.out.println("Tau=" + MatrixOps.arrToStr(Tau));
		System.out.println("lambda=" + MatrixOps.doubleArrayToPrintString(Lambda));
		System.out.println("betas=" + MatrixOps.doubleArrayToPrintString(betas));
		System.out.println();
		try {
			Thread.sleep(1000);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
*/
	}

}
