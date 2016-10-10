package xyz.lejon.bayes.models.probit;

import static java.lang.Math.abs;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;
import static org.ejml.ops.CommonOps.addEquals;
import static org.ejml.ops.CommonOps.invert;
import static org.ejml.ops.CommonOps.multTransA;
import static xyz.lejon.utils.MatrixOps.pow;
import static xyz.lejon.utils.MatrixOps.rnorm;
import static xyz.lejon.utils.MatrixOps.scalarDivide;
import static xyz.lejon.utils.MatrixOps.scalarMultiply;
import static xyz.lejon.utils.MatrixOps.scalarPlus;
import static xyz.lejon.utils.MatrixOps.sqrt;
import static xyz.lejon.utils.MatrixOps.sum;

import java.io.IOException;
import java.util.Arrays;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import xyz.lejon.configuration.DOConfiguration;
import xyz.lejon.sampling.Gamma;
import xyz.lejon.utils.MatrixOps;

public class HorseshoeDOProbitPX extends SerialDOSampler {

	MatrixOps mo = new MatrixOps();
	double [][] Theta;
	double Sigma = 1.0;
	double G = 1.0;
	double Delta = 0.1;
	double [] Tau;
	double [][] Lambda;
	double [][] precMtrx;
	
	public HorseshoeDOProbitPX(DOConfiguration config, double[][] xs, int[] ys, int noClasses) throws IOException {
		super(config, xs, ys, noClasses);
		Theta = new double[noClasses][];
		for (int i = 0; i < Theta.length; i++) {
			Theta[i] = new double[noCovariates];
			Arrays.fill(Theta[i], 1.0);
		}
		Lambda = new double[noClasses][];
		for (int i = 0; i < Lambda.length; i++) {
			Lambda[i] = new double[noCovariates];
			Arrays.fill(Lambda[i], 1.0);
		}
		Tau = new double[noClasses];
		for (int i = 0; i < Tau.length; i++) {
			Tau[i] = Math.abs(Delta)*G;
		}
		precMtrx = new double[Xd.numCols][Xd.numCols];
		for (int i = 0; i < noClasses; i++) {
			for (int j = 0; j < noCovariates; j++) {
				betas[i][j] = rnorm(0.0, 1.0);
			}
		}
	}
	
	public double [] calcTheta(double[] Beta, double Sigma, double Delta, double[] Lambda) {
		double [] Theta = scalarDivide(Beta,scalarMultiply(Lambda,Sigma*Delta));
		return Theta;
	}
	
	public double sampleSigma2(double [][] Res2, double Tau, double[] Lambda, double [][] Y) {
		// Jeffreys prior is assumed
		// for(i in 1:n)
		//  {
		//    Res2[,i] = {Y[,i]^2}/{1+(Tau^2)*{Lambda^2}}
		//  }
		  
		for (int row = 0; row < Res2.length; row++) {			
			for(int i=0; i < noCovariates; i++)
			{
				Res2[row][i] = pow(Y[row][i],2) / (1 + pow(Tau,2) * pow(Lambda[i],2) );
			}
		}
		// RSS = sum(Res2)
		double RSS = mo.sumPar(Res2);
		// Sigma2 = 1/rgamma(1,n*p/2, rate = RSS/2)
		double Sigma2 = 1/Gamma.rgamma(1,noRows*noCovariates/2.0, RSS/2);
		// Sigma = sqrt(Sigma2)
		
		return Sigma2;
	}

	public double [] sampleTau(double Tau, double [] Theta, double [] Ybar, double Sigma, double [] Lambda) {
		//G = 1/sqrt(rgamma(1,(p+1)/2, rate = (1+sum(Theta^2))/2))
		double G = 1/sqrt(Gamma.rgamma(1, (noCovariates+1)/2, (1+sum(pow(Theta,2)))/2));
		//Z = Ybar/(Sigma*Theta*Lambda)
		double [] Z = scalarDivide(Ybar, scalarMultiply(scalarMultiply(Theta, Sigma),Lambda));
		//a = n*(Lambda*Theta)^2
		double [] a = scalarMultiply(pow(scalarMultiply(Lambda,Theta),2),noRows);
		//b = sum(a)
		double b = sum(a);
		//s2 = 1/(1+b)
		double s2 = 1/(1+b);
		//m = {s2}*sum(a*Z)
		double m = s2*sum(scalarMultiply(a,Z));
		//Delta = rnorm(1,m,sqrt(s2))
		double Delta = rnorm(m,sqrt(s2));
		//Tau = abs(Delta)*G
		double newTau = abs(Delta)*G;

		double [] res = {newTau, Delta};
		return res;
	}

	public double [] sampleLambda(double Tau, double [] Theta, double [] Ybar, 
			double Sigma, double [] Lambda, double Delta) {
		//Z = Ybar/(Sigma*Delta*Theta)
		double [] Z = scalarDivide(Ybar, scalarMultiply(scalarMultiply(Theta, Sigma),Delta));
		//V2 = 1/rgamma(p,1,rate=(Lambda^2+1)/2)
		double [] V2 = scalarDivide(1,Gamma.rgamma(noCovariates,1.0,scalarDivide(scalarPlus(pow(Lambda,2),1),2)));
		//num1 = n*V2*((Delta*Theta)^2)
		double [] num1 = scalarMultiply(scalarMultiply(V2,(pow(scalarMultiply(Theta,Delta),2))),noRows);
		//den = 1 + num1
		double [] den = scalarPlus(num1,1);
		//s = sqrt(V2/den)
		double [] s = sqrt(scalarDivide(V2,den));
		//m = {num1/den}*Z
		double [] m = scalarMultiply(scalarDivide(num1,den),Z);
		//Lambda = rnorm(p,m,s)
		double [] newLambda = rnorm(noCovariates,m,s);

		return newLambda;
	}

	public double [] sampleBeta(double Tau, double [] Lambda, double Sigma2, double [] Ybar) {
		// a = (Tau^2)*(Lambda^2)
		double [] a = scalarMultiply(pow(Lambda,2),pow(Tau,2));
		// b = n*a
		double [] b = scalarMultiply(a,noRows);
		// s = sqrt(Sigma2*a/{1+b})
		double [] s = sqrt(scalarDivide(scalarMultiply(a,Sigma2),scalarPlus(b,1.0)));
		// m = {b/{1+b}}*Ybar
		double [] m = scalarMultiply(scalarDivide(b,scalarPlus(b,1)),Ybar);
		// Beta = rnorm(p, m, s)
		double [] Beta = rnorm(noCovariates, m, s);
		
		return Beta;
	}
	
	@Override
	public void sampleBeta(int k) {
		double [] zColk = zsT[k];
		
		Theta[k]  = calcTheta(betas[k], Sigma, Delta, Lambda[k]);
		//System.out.println("Theta: " + MatrixOps.doubleArrayToPrintString(Theta));
		double [] res = sampleTau(Tau[k], Theta[k], betas[k], Sigma, Lambda[k]);
		Tau[k]    = res[0];
		Delta     = res[1];
		System.out.println(MatrixOps.arrToStr(Tau, "Tau"));
		Lambda[k] = sampleLambda(Tau[k], Theta[k], betas[k], Sigma, Lambda[k], Delta);
		//DenseMatrix64F td = new DenseMatrix64F(Lambda);
		//System.out.println("New Lambda: " + td);

		// Rebuild precision matrix
		for (int i = 0; i < precMtrx.length; i++) {
			for (int j = 0; j < precMtrx[i].length; j++) {
				if(i==j) {
					precMtrx[i][j] = Math.pow(Tau[k], 2) * Math.pow(Lambda[k][i],2);
				}
			}
		}
		//System.out.println(MatrixOps.doubleArrayToPrintString(precMtrx));
		priorPrecision = new DenseMatrix64F(precMtrx);
		
		priorMean = new DenseMatrix64F(XtX);
		addEquals(priorMean, priorPrecision);
		invert(priorMean);
		Stilde = MatrixOps.extractDoubleArray(priorMean);
		
		DenseMatrix64F zColKd = new DenseMatrix64F(zColk.length,1);
		zColKd.setData(zColk);
		DenseMatrix64F localTilde = new DenseMatrix64F(muTilde);
		DenseMatrix64F localMu = new DenseMatrix64F(Xd.numCols, 1);
		multTransA(Xd, zColKd, localTilde);
		CommonOps.mult(priorMean, localTilde, localMu);
		double [] mu_tile = localMu.getData();

		MultivariateNormalDistribution mvn = new MultivariateNormalDistribution(mu_tile, Stilde);
		betas[k] = mvn.sample();
	}
	
	public void sampleBetaOLD(int k) {
		
		Theta[k]  = calcTheta(betas[k], Sigma, Delta, Lambda[k]);
		//System.out.println("Theta: " + MatrixOps.doubleArrayToPrintString(Theta));
		double [] res = sampleTau(Tau[k], Theta[k], betas[k], Sigma, Lambda[k]);
		Tau[k]    = res[0];
		Delta     = res[1];
		//System.out.println(MatrixOps.arrToStr(Tau, "Tau"));
		Lambda[k] = sampleLambda(Tau[k], Theta[k], betas[k], Sigma, Lambda[k], Delta);
		//DenseMatrix64F td = new DenseMatrix64F(Lambda);
		//System.out.println("New Lambda: " + td);

		// Rebuild precision matrix
		for (int i = 0; i < precMtrx.length; i++) {
			for (int j = 0; j < precMtrx[i].length; j++) {
				if(i==j) {
					precMtrx[i][j] = Math.pow(Tau[k], 2) * Math.pow(Lambda[k][i],2);
				}
			}
		}
		//System.out.println(MatrixOps.doubleArrayToPrintString(precMtrx));
		priorPrecision = new DenseMatrix64F(precMtrx);
		
		DenseMatrix64F XtXA = new DenseMatrix64F(XtX);
		addEquals(XtXA, priorPrecision);
		invert(XtXA);
		Stilde = MatrixOps.extractDoubleArray(XtXA);
		//System.out.println(XtXA);
		DenseMatrix64F mumuk = new DenseMatrix64F(Xd.numCols, 1);
		
		double [] zColk = zsT[k];
		DenseMatrix64F ak = new DenseMatrix64F(zColk.length,1);
		ak.setData(zColk);
		CommonOps.multAddTransA(Xd, ak, mumuk);
		DenseMatrix64F mud = new DenseMatrix64F(Xd.numCols,1);
		//System.out.println("XtXA: " +XtXA);
		//System.out.println("mumuk:" + mumuk);		
		CommonOps.multAdd(XtXA, mumuk, mud);
		//System.out.println(mud);
		double [] mu = MatrixOps.extractDoubleVector(mud);

		MultivariateNormalDistribution mvn = new MultivariateNormalDistribution(mu, Stilde);
		betas[k] = mvn.sample();
	}
}
