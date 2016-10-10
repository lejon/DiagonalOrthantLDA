package xyz.lejon.bayes.models.probit;

import static org.junit.Assert.assertTrue;

import java.io.IOException;

import joinery.DataFrame;
import joinery.DataFrame.NumberDefault;

import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import xyz.lejon.bayes.models.regression.LinearRegressionHSPrior;
import xyz.lejon.bayes.models.regression.LinearRegressionJBlasHSPrior;
import xyz.lejon.utils.MatrixOps;

public class HorseshoeTest {

	KolmogorovSmirnovTest ks = new KolmogorovSmirnovTest();

	int size = 5000;
	double [] mu_n_1 = {-1.51, 0.39, 1.96};
	double [][] Lambda_n_inv_1 = MatrixOps.diag(3,1.0/5);
	double sigma_1 = 1;

	double [] mu_n_2 = {-0.52, -1.49};
	double [][] Lambda_n_inv_2 = MatrixOps.diag(2);
	double sigma_2 = 0.1;

	double a_n_1 = 172;
	double b_0_1 = 1;
	double yty_1 = 344;

	double a_n_2 = 12;
	double b_0_2 = 5;
	double yty_2 = 36;

	double [] lambda_1 = {1,1,1};
	double [] beta_1 = {1,-1,2};
	double tau_1 = 1;

	double [] lambda_2 = {0.5,2,1,3};
	double [] beta_2 = {0.1,2,-1,4};
	double tau_2 = 0.1;
	String sep=",";
	String naString = "NA";
	boolean hasHeader = true;
	double p = 0.00001;

	@Test
	public void testDrawTau1() throws IOException {
		String expectedFn = "src/test/resources/horseshoe/dftau.csv";
		double [] df_tau = new double[size];
		DataFrame<Object> df = DataFrame.readCsv(expectedFn, sep, NumberDefault.DOUBLE_DEFAULT, naString, hasHeader);
		DataFrame<Double> ddf = df.cast(Double.class);
		double [][] xstmp = ddf.transpose().fillna(0.0).toArray(double[][].class);
		double [] expected = xstmp[1];
		for(int i = 0; i < size; i++){
			df_tau[i] = HorseshoeDOProbit.sampleTau(tau_1,beta_1,lambda_1,sigma_1, false);
		}
		double kolmogorovSmirnovTestResult = ks.kolmogorovSmirnovTest(expected, df_tau);
		/*System.out.println("Draws:");
		for (int i = 0; i < expected.length; i++) {
			System.out.println(expected[i]  + "\t<=> " + df_tau[i]);
		}
		System.out.println();*/
		assertTrue("Kolmogorov don't think they are the same! " + kolmogorovSmirnovTestResult + " !> " + p, kolmogorovSmirnovTestResult > p);
		System.out.println("testDrawTau1 OK! Kolmogorow: " + kolmogorovSmirnovTestResult);
	}
	
	@Test
	public void testDrawTau2() throws IOException {
		String expectedFn = "src/test/resources/horseshoe/dftau.csv";
		double [] df_tau = new double[size];
		DataFrame<Object> df = DataFrame.readCsv(expectedFn, sep, NumberDefault.DOUBLE_DEFAULT, naString, hasHeader);
		DataFrame<Double> ddf = df.cast(Double.class);
		double [][] xstmp = ddf.transpose().fillna(0.0).toArray(double[][].class);
		double [] expected = xstmp[2];
		for(int i = 0; i < size; i++){
			df_tau[i] = HorseshoeDOProbit.sampleTau(tau_2,beta_2,lambda_2,sigma_2, false);
		}
		double kolmogorovSmirnovTestResult = ks.kolmogorovSmirnovTest(expected, df_tau);
		/*System.out.println("Draws:");
		for (int i = 0; i < expected.length; i++) {
			System.out.println(expected[i]  + "\t<=> " + df_tau[i]);
		}
		System.out.println();*/
		assertTrue("Kolmogorov don't think they are the same! " + kolmogorovSmirnovTestResult + " !> " + p, kolmogorovSmirnovTestResult > p);
		System.out.println("testDrawTau2 OK! Kolmogorow: " + kolmogorovSmirnovTestResult);
	}
	
	@Test
	public void testDrawLambda1() throws IOException {
		String expectedFn = "src/test/resources/horseshoe/dflambda1.csv";
		DataFrame<Object> df = DataFrame.readCsv(expectedFn, sep, NumberDefault.DOUBLE_DEFAULT, naString, hasHeader);
		DataFrame<Double> ddf = df.drop(0).cast(Double.class);
		double [][] expected = ddf.toArray(double[][].class);
		double [][] df_lambda = new double[expected.length][expected[0].length];
		
		for(int i = 0; i < expected.length; i++){
			df_lambda[i] = HorseshoeDOProbit.sampleLambda(tau_1,beta_1,lambda_1,sigma_1);
		}
		
		for (int covariate = 0; covariate < expected[0].length; covariate++) {
			double[] expecetedCovariateSamples = MatrixOps.extractColVector(covariate,	expected);
			double[] covariateSamples = MatrixOps.extractColVector(covariate,df_lambda);
			double kolmogorovSmirnovTestResult = ks.kolmogorovSmirnovTest(expecetedCovariateSamples, covariateSamples);
			/*System.out.println("Draws:");
			for (int i = 0; i < expecetedCovariateSamples.length; i++) {
				System.out.println(expecetedCovariateSamples[i]  + "\t<=> " + covariateSamples[i]);
			}
			System.out.println();*/
			assertTrue("Kolmogorov don't think covariate " 
					+ covariate 
					+ " are the same! " 
					+ kolmogorovSmirnovTestResult + " !> " + p, 
					kolmogorovSmirnovTestResult > p);
			System.out.println("testDrawLambda1 covariate " + covariate + " OK! Kolmogorow: " + kolmogorovSmirnovTestResult);
		}
	}
	
	@Test
	public void testDrawLambda2() throws IOException {
		String expectedFn = "src/test/resources/horseshoe/dflambda2.csv";
		DataFrame<Object> df = DataFrame.readCsv(expectedFn, sep, NumberDefault.DOUBLE_DEFAULT, naString, hasHeader);
		DataFrame<Double> ddf = df.drop(0).cast(Double.class);
		double [][] expected = ddf.toArray(double[][].class);
		double [][] df_lambda = new double[expected.length][expected[0].length];
		
		for(int i = 0; i < expected.length; i++){
			df_lambda[i] = HorseshoeDOProbit.sampleLambda(tau_2,beta_2,lambda_2,sigma_2);
		}
		
		for (int covariate = 0; covariate < expected[0].length; covariate++) {
			double[] expecetedCovariateSamples = MatrixOps.extractColVector(covariate,	expected);
			double[] covariateSamples = MatrixOps.extractColVector(covariate,df_lambda);
			double kolmogorovSmirnovTestResult = ks.kolmogorovSmirnovTest(expecetedCovariateSamples, covariateSamples);
			/*System.out.println("Draws:");
			for (int i = 0; i < expecetedCovariateSamples.length; i++) {
				System.out.println(expecetedCovariateSamples[i]  + "\t<=> " + covariateSamples[i]);
			}
			System.out.println();*/
			assertTrue("Kolmogorov don't think covariate " 
					+ covariate 
					+ " are the same! " 
					+ kolmogorovSmirnovTestResult + " !> " + p, 
					kolmogorovSmirnovTestResult > p);
			System.out.println("testDrawLambda1 covariate " + covariate + " OK! Kolmogorow: " + kolmogorovSmirnovTestResult);
		}
	}
	
	@Test
	public void testDrawSigma1() throws IOException {
		String expectedFn = "src/test/resources/horseshoe/dfsigma.csv";
		double [] df_sigma = new double[size];
		DataFrame<Object> df = DataFrame.readCsv(expectedFn, sep, NumberDefault.DOUBLE_DEFAULT, naString, hasHeader);
		DataFrame<Double> ddf = df.cast(Double.class);
		double [][] xstmp = ddf.transpose().fillna(0.0).toArray(double[][].class);
		double [] expected = xstmp[1];
		double [][] yty1 = {{yty_1}};
		DoubleMatrix yty = new DoubleMatrix(yty1);
		for(int i = 0; i < size; i++){
			df_sigma[i] = Math.sqrt(LinearRegressionJBlasHSPrior.sampleSigmaSquared(new DoubleMatrix(mu_n_1), new DoubleMatrix(Lambda_n_inv_1), yty, (int) (2 * (a_n_1-LinearRegressionHSPrior.v0))));
		}
		double kolmogorovSmirnovTestResult = ks.kolmogorovSmirnovTest(expected, df_sigma);
		/*System.out.println("Draws:");
		for (int i = 0; i < expected.length; i++) {
			System.out.println(expected[i]  + "\t<=> " + df_sigma[i]);
		}
		System.out.println();*/
		assertTrue("Kolmogorov don't think they are the same! " + kolmogorovSmirnovTestResult + " !> " + p, kolmogorovSmirnovTestResult > p);
		System.out.println("testDrawSigma1 OK! Kolmogorow: " + kolmogorovSmirnovTestResult);
	}
	
	@Test
	public void testDrawSigma2() throws IOException {
		String expectedFn = "src/test/resources/horseshoe/dfsigma.csv";
		double [] df_sigma = new double[size];
		DataFrame<Object> df = DataFrame.readCsv(expectedFn, sep, NumberDefault.DOUBLE_DEFAULT, naString, hasHeader);
		DataFrame<Double> ddf = df.cast(Double.class);
		double [][] xstmp = ddf.transpose().fillna(0.0).toArray(double[][].class);
		double [] expected = xstmp[2];
		double [][] yty1 = {{yty_2}};
		DoubleMatrix yty = new DoubleMatrix(yty1);
		LinearRegressionHSPrior.tau0 = 5;
		for(int i = 0; i < size; i++){
			df_sigma[i] = Math.sqrt(LinearRegressionJBlasHSPrior.sampleSigmaSquared(new DoubleMatrix(mu_n_2), new DoubleMatrix(Lambda_n_inv_2), yty, (int) (2 * (a_n_2-LinearRegressionHSPrior.v0))));
		}
		double kolmogorovSmirnovTestResult = ks.kolmogorovSmirnovTest(expected, df_sigma);
		/*System.out.println("Draws:");
		for (int i = 0; i < expected.length; i++) {
			System.out.println(expected[i]  + "\t<=> " + df_sigma[i]);
		}
		System.out.println();*/
		assertTrue("Kolmogorov don't think they are the same! " + kolmogorovSmirnovTestResult + " !> " + p, kolmogorovSmirnovTestResult > p);
		System.out.println("testDrawSigma2 OK! Kolmogorow: " + kolmogorovSmirnovTestResult);
	}
	
	@Test
	public void testDrawBeta1() throws IOException {
		String expectedFn = "src/test/resources/horseshoe/dfbeta1.csv";
		DataFrame<Object> df = DataFrame.readCsv(expectedFn, sep, NumberDefault.DOUBLE_DEFAULT, naString, hasHeader);
		DataFrame<Double> ddf = df.drop(0).cast(Double.class);
		double [][] expected = ddf.toArray(double[][].class);
		double [][] df_beta = new double[expected.length][expected[0].length];
		
		for(int i = 0; i < expected.length; i++){
			df_beta[i] = LinearRegressionJBlasHSPrior.sampleBeta(new DoubleMatrix(mu_n_1), new DoubleMatrix(Lambda_n_inv_1), sigma_1);
		}
		
		for (int covariate = 0; covariate < expected[0].length; covariate++) {
			double[] expecetedCovariateSamples = MatrixOps.extractColVector(covariate,	expected);
			double[] covariateSamples = MatrixOps.extractColVector(covariate,df_beta);
			double kolmogorovSmirnovTestResult = ks.kolmogorovSmirnovTest(expecetedCovariateSamples, covariateSamples);
			/*System.out.println("Draws:");
			for (int i = 0; i < expecetedCovariateSamples.length; i++) {
				System.out.println(expecetedCovariateSamples[i]  + "\t<=> " + covariateSamples[i]);
			}
			System.out.println();*/
			assertTrue("Kolmogorov don't think covariate " 
					+ covariate 
					+ " are the same! " 
					+ kolmogorovSmirnovTestResult + " !> " + p, 
					kolmogorovSmirnovTestResult > p);
			System.out.println("testDrawBeta1 covariate " + covariate + " OK! Kolmogorow: " + kolmogorovSmirnovTestResult);
		}
	}
	
	@Test
	public void testDrawBeta2() throws IOException {
		String expectedFn = "src/test/resources/horseshoe/dfbeta2.csv";
		DataFrame<Object> df = DataFrame.readCsv(expectedFn, sep, NumberDefault.DOUBLE_DEFAULT, naString, hasHeader);
		DataFrame<Double> ddf = df.drop(0).cast(Double.class);
		System.out.println("Expect: " + ddf);
		double [][] expected = ddf.toArray(double[][].class);
		double [][] df_beta = new double[expected.length][expected[0].length];
		
		for(int i = 0; i < expected.length; i++){
			df_beta[i] = LinearRegressionJBlasHSPrior.sampleBeta(new DoubleMatrix(mu_n_2), new DoubleMatrix(Lambda_n_inv_2), sigma_2);
		}
		
		for (int covariate = 0; covariate < expected[0].length; covariate++) {
			double[] expecetedCovariateSamples = MatrixOps.extractColVector(covariate,	expected);
			double[] covariateSamples = MatrixOps.extractColVector(covariate,df_beta);
			double kolmogorovSmirnovTestResult = ks.kolmogorovSmirnovTest(expecetedCovariateSamples, covariateSamples);
			/*System.out.println("Draws:");
			for (int i = 0; i < expecetedCovariateSamples.length; i++) {
				System.out.println(expecetedCovariateSamples[i]  + "\t<=> " + covariateSamples[i]);
			}
			System.out.println();*/
			assertTrue("Kolmogorov don't think covariate " 
					+ covariate 
					+ " are the same! " 
					+ kolmogorovSmirnovTestResult + " !> " + p, 
					kolmogorovSmirnovTestResult > p);
			System.out.println("testDrawBeta2 covariate " + covariate + " OK! Kolmogorow: " + kolmogorovSmirnovTestResult);
		}
	}
}
