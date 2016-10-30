package xyz.lejon.bayes.models.regression;

import static org.ejml.ops.CommonOps.invert;
import static org.ejml.ops.CommonOps.multTransA;

import java.io.IOException;
import java.util.Arrays;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import xyz.lejon.bayes.models.probit.HorseshoeDOProbit;
import xyz.lejon.configuration.OLSConfiguration;
import xyz.lejon.sampling.BasicRDists;
import xyz.lejon.sampling.FastMVNSamplerEJML;
import xyz.lejon.sampling.InverseGammaDistribution;
import xyz.lejon.utils.MatrixOps;

/**
 * Multiple Bayesian Linear Regression with a horseshoe prior on the betas 
 * 
 * @author Leif Jonsson
 *
 */
public class LinearRegressionEJMLHSPrior extends LinearRegressionHSPrior {
	protected DenseMatrix64F Xd;
	protected DenseMatrix64F XtX;
	protected DenseMatrix64F muTilde;
	protected DenseMatrix64F ys;
	protected DenseMatrix64F yty;
	protected DenseMatrix64F myPrecision;
	
	public LinearRegressionEJMLHSPrior(OLSConfiguration config, double[][] xs, double[] ys) throws IOException {
		super(config, xs, ys);
		Lambda = new double[noCovariates];
		Arrays.fill(Lambda, 1.0); 
		for (int j = 0; j < noCovariates; j++) {
			betas[j] = BasicRDists.rnorm(0.0, 1.0);
		}
		
		this.ys = new DenseMatrix64F(ys.length,1);
		this.ys.setData(ys.clone());
		
		Xd = new DenseMatrix64F(xs);
		XtX = new DenseMatrix64F(Xd.numCols,Xd.numCols);
		CommonOps.multTransA(Xd,Xd,XtX);
		yty = new DenseMatrix64F(this.ys.numCols,this.ys.numCols);
		CommonOps.multTransA(this.ys,this.ys,yty);
			
		myPrecision = XtX.copy();
		
		System.out.println("################# Running LinearRegressionEJMLHSPrior #################");
	}
	
	public static double sampleSigmaSquared(DenseMatrix64F mu, DenseMatrix64F precision, DenseMatrix64F yty, int noObservations) {
		double vn = v0  + noObservations / (double) 2;

		DenseMatrix64F tmp1 = new DenseMatrix64F(precision.numRows,mu.numCols);
		CommonOps.mult(precision,mu,tmp1);
		DenseMatrix64F tmp2 = new DenseMatrix64F(mu.numCols,tmp1.numCols);
		CommonOps.multTransA(mu, tmp1, tmp2);
		DenseMatrix64F tmp3 = new DenseMatrix64F(yty.numCols,yty.numRows);
		CommonOps.subtract(yty,tmp2,tmp3);
		
		if(tmp3.numRows!=1 && tmp3.numCols!=1) {
			throw new IllegalStateException("intermediary has wrong dims!");
		}
		
		double taun = tau0 + tmp3.get(0) / 2;
		
		InverseGammaDistribution ig = new InverseGammaDistribution(vn, taun);

		return ig.nextInverseGamma();
	}
	
	@Override
	public double sampleSigmaSquared(Object muIn, Object precisionIn) {
		DenseMatrix64F mu = (DenseMatrix64F) muIn;
		DenseMatrix64F precision = (DenseMatrix64F) precisionIn;

		return sampleSigmaSquared(mu,precision,yty,noRows);
	}
	
	@Override
	public void sampleBeta() {
		Tau    = HorseshoeDOProbit.sampleTau(Tau, betas, Lambda, Sigma, useIntercept);
		Lambda = HorseshoeDOProbit.sampleLambda(Tau, betas, Lambda, Sigma);
				
		sampledTaus.add(Tau);
		sampledLambdas.add(Lambda.clone());

		// Rebuild updated precision matrix
		for (int i = 0; i < myPrecision.numRows; i++) {
			double update;
			// Don't regularize the intercept
			if(i==0 && useIntercept) {
				update = c;
			} else {
				update = Math.pow(Tau, 2) * Math.pow(Lambda[i],2);
			}
			myPrecision.set(i, i, XtX.get(i,i)  + 1 / update);
		}

		DenseMatrix64F pInv = myPrecision.copy();
		invert(pInv);

		DenseMatrix64F localTilde = new DenseMatrix64F(Xd.numCols,ys.numCols);
		multTransA(Xd, ys, localTilde);
		DenseMatrix64F localMu = new DenseMatrix64F(Xd.numCols, 1);
		CommonOps.mult(pInv, localTilde, localMu);
		
		for (int i = 0; i < pInv.numRows; i++) {
			for (int j = 0; j < pInv.numCols; j++) {				
				if(Double.isNaN(pInv.get(i,i))) {
					System.err.println(
							"Covariance matrix is broken"
									+ " betas=" + MatrixOps.arrToStr(betas) + "\n"
									+ " Tau=" + Tau + "\n"
									+ " Lambda=" + MatrixOps.arrToStr(Lambda) + "\n"
							);
					System.exit(-1);
				}
			}
		}

		betas = sampleBeta(localMu, pInv, Sigma);
		
		Sigma  = Math.sqrt(sampleSigmaSquared(localMu, myPrecision));
	}

	public static double [] sampleBeta(DenseMatrix64F localMu, DenseMatrix64F pInv, double sigma) {
		CommonOps.scale(Math.pow(sigma,2), pInv);
		FastMVNSamplerEJML mvn = new FastMVNSamplerEJML(localMu, pInv);
		double [] betas = mvn.sample(localMu);
		for (int i = 0; i < betas.length; i++) {
			if(Double.isNaN(betas[i])) {
				throw new IllegalArgumentException(
						"Sampled invalid beta:"
						+ " betas=" + MatrixOps.arrToStr(betas) + "\n"
						);
			}
		}
		return betas;
	}
}
