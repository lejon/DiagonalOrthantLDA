package xyz.lejon.bayes.models.dolda;

import static org.ejml.ops.CommonOps.invert;
import static org.ejml.ops.CommonOps.multInner;
import static org.ejml.ops.CommonOps.multTransA;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.Arrays;

import org.apache.commons.cli.ParseException;
import org.apache.commons.configuration.ConfigurationException;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;
import org.junit.Test;

import xyz.lejon.bayes.models.probit.DOEvaluation;
import xyz.lejon.configuration.ConfigFactory;
import xyz.lejon.configuration.DOConfiguration;
import xyz.lejon.configuration.DOLDACommandLineParser;
import xyz.lejon.configuration.DOLDAConfiguration;
import xyz.lejon.eval.EvalResult;
import xyz.lejon.sampling.FastMVNSamplerJBLAS;
import xyz.lejon.sampling.FastMultivariateNormalDistribution;
import xyz.lejon.utils.BlasOps;
import xyz.lejon.utils.LoggingUtils;
import xyz.lejon.utils.MatrixOps;

public class BetaSamplingTest {
	
	double [][] Ast = {{1,2,3},{4,5,6}};
	double [][] Xs = {{1,2,3},{4,5,6}, {7,8,9}};
	int ks = 1;
	int p = 2;
	double [] Tau = {1,2,3};
	double [][] Lambda = {{0.1,0.2,0.3}, {0.4,0.5,0.6}};
	double [][] betas = {{10,11,12}, {13,14,15}};
	double Sigma = 1.0;
	int k = 0;
	int noClasses = 2;
	
	@Test
	public void testEJML() {
		double [][] betasTmp = new double [2][];
		for (int i = 0; i < betasTmp.length; i++) {
			betasTmp[i] = Arrays.copyOf(betas[i], betas[i].length);
		}
		for (int k = 0; k < noClasses; k++) {			
			sampleBetaEJML(Arrays.copyOf(Xs, Xs.length), k);
			System.out.println("\t ===========> Sampled betas: " + MatrixOps.arrToStr(betas[k]));
		}
		for (int i = 0; i < betasTmp.length; i++) {
			betas[i] = Arrays.copyOf(betasTmp[i], betasTmp[i].length);
		}
	}

	@Test
	public void testJBlas() {
		double [][] betasTmp = new double [2][];
		for (int i = 0; i < betasTmp.length; i++) {
			betasTmp[i] = Arrays.copyOf(betas[i], betas[i].length);
		}
		for (int k = 0; k < noClasses; k++) {	
			sampleBetaJBlas(Arrays.copyOf(Xs, Xs.length), k);
			System.out.println("\t ===========> Sampled betas: " + MatrixOps.arrToStr(betas[k]));
		}
		for (int i = 0; i < betasTmp.length; i++) {
			betas[i] = Arrays.copyOf(betasTmp[i], betasTmp[i].length);
		}
	}

	
	void sampleBetaEJML(double [][] Xs, int k) {		
		DenseMatrix64F X  = new DenseMatrix64F(Xs);
		DenseMatrix64F XtX = new DenseMatrix64F(X.numCols, X.numCols);
		DenseMatrix64F precision = new DenseMatrix64F(ks+p,ks+p);
		
		multInner(X, XtX);
		
		// Rebuild updated precision matrix
		for (int i = 0; i < precision.numRows; i++) {
			for (int j = 0; j < precision.numCols; j++) {
				if(i==j) {
					double update = (Math.pow(Tau[k], 2) * Math.pow(Lambda[k][i],2));
					precision.set(i, j, XtX.get(i,j) + 1 / update);
				} else {
					precision.set(i, j, XtX.get(i,j));
				}
			}
		}
		
		System.out.println("EJML initial precision: " + MatrixOps.doubleArrayToPrintString(MatrixOps.extractDoubleArray(precision)));

		invert(precision);

		double [] zColk = Ast[k];
		DenseMatrix64F zColKd = new DenseMatrix64F(zColk.length,1);
		zColKd.setData(zColk);
		DenseMatrix64F localTilde = new DenseMatrix64F(X.numCols, 1);
		DenseMatrix64F localMu = new DenseMatrix64F(X.numCols, 1);
		multTransA(X, zColKd, localTilde);
		CommonOps.mult(precision, localTilde, localMu);

		invert(precision);
		
		System.out.println("EJML local mu: " + MatrixOps.arrToStr(localMu.getData()));
		System.out.println("EJML precision: " + MatrixOps.doubleArrayToPrintString(MatrixOps.extractDoubleArray(precision)));

		FastMultivariateNormalDistribution amvn = new FastMultivariateNormalDistribution(localMu.data, MatrixOps.extractDoubleArray(precision));
		//FastMVNSampler mvns = new FastMVNSampler(localMu, precision);
		betas[k] = amvn.sample();
		
		sampleTauAndLambda(k);
	}
	
	private void sampleBetaJBlas(double [][] Xs, int k) {		
		DoubleMatrix X = new DoubleMatrix(Xs);
		DoubleMatrix XtX = X.transpose().mmul(X);
		DoubleMatrix Xt = new DoubleMatrix();
		DoubleMatrix precision = new DoubleMatrix();
	
		Xt.copy(X);
		Xt = Xt.transpose();

		precision = new DoubleMatrix(ks+p,ks+p);
		// Build precision matrix
		for (int i = 0; i < precision.rows; i++) {
			for (int j = 0; j < precision.columns; j++) {
				if(i==j) {
					double update = (Math.pow(Tau[k], 2) * Math.pow(Lambda[k][i],2));
					precision.put(i, j, XtX.get(i, j) + 1 / update);
				} else {
					precision.put(i, j, XtX.get(i, j));
				}
			}
		}
		
		System.out.println("JBLAS initial precision: " + MatrixOps.doubleArrayToPrintString(precision.toArray2()));

		double [] zColk = Ast[k];
		DoubleMatrix zColKd = new DoubleMatrix(zColk);
		DoubleMatrix localTilde = Xt.mmul(zColKd);
		DoubleMatrix localMu = new DoubleMatrix(X.columns, 1);

		localMu = Solve.solve(precision, localTilde);
		DoubleMatrix pInv = BlasOps.blasInvert(precision);
		//localMu = pInv.mmul(localTilde);
		
		// We don't have to multiply with Sigma here since it is always 1
		FastMVNSamplerJBLAS mvn = new FastMVNSamplerJBLAS(pInv);
		//FastMVNSamplerJBLAS mvn = new FastMVNSamplerJBLAS(precision, true);
		
		System.out.println("JBLAS local mu: " + MatrixOps.arrToStr(localMu.data));
		System.out.println("JBLAS precision: " + MatrixOps.doubleArrayToPrintString(precision.toArray2()));
		betas[k] = mvn.sample(localMu);
		
		sampleTauAndLambda(k);
	}
	
	synchronized void sampleTauAndLambda(int k) {
	}


	@Test
	public void testSmokeSerialWithTestset() throws ParseException, ConfigurationException, IOException {
		ConfigFactory.resetFactory();
		String [] args = {"--run_cfg=src/main/resources/configuration/DOLDABasicTest.cfg"};
		DOLDACommandLineParser cp = new DOLDACommandLineParser(args);
		DOLDAConfiguration config = (DOLDAConfiguration) ConfigFactory.getMainConfiguration(cp);

		String [] configs = config.getSubConfigs();
		for(String conf : configs) {
			config.activateSubconfig(conf);
			LoggingUtils lu = new LoggingUtils();
			String logSuitePath = "Runs/RunSuite" + LoggingUtils.getDateStamp();
			lu.checkAndCreateCurrentLogDir(logSuitePath);

			config.setLoggingUtil(lu);
			DOLDADataSet trainingSetData =config.loadCombinedTrainingSet();
			DOLDADataSet testSetData = config.loadCombinedTestSet();

			double [][] xs = trainingSetData.getX();
			int [] ys = trainingSetData.getY();

			DOLDA dolda  = new DOLDAGibbsJBlasHorseshoe(config, xs, ys);
			dolda.sample(config.getNoIterations(DOConfiguration.ITERATIONS_DEFAULT));

			double [][] betas = dolda.getBetas();
			
			double [][] testset;
			int [] testLabels;
			if(testSetData==null) {
				testset = xs;
				testLabels = ys;
			} else {
				System.out.println("Using testset!");
				testset = testSetData.getX();
				testLabels = testSetData.getY();
			}

			EvalResult result  = DOEvaluation.evaluate(testset, testLabels, betas);
			double pCorrect = ((double)result.noCorrect)/testset.length * 100;
			System.out.println("Accuracy: " + pCorrect);
			assertTrue(pCorrect>0.75);
		}
	}
	
}
