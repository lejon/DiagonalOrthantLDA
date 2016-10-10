package xyz.lejon.bayes.models.dolda;

import static java.lang.Math.exp;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

import joinery.DataFrame;
import joinery.DataFrame.NumberDefault;

import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.math3.stat.inference.TTest;
import org.junit.Test;

import xyz.lejon.utils.MatrixOps;
import cc.mallet.util.LDAUtils;

public class ZSamplingTest {
	int it = 5000;
	String sep=",";
	String naString = "NA";
	boolean hasHeader = true;

	double p_thr = 0.01;

	@Test
	public void testZSampling() throws ConfigurationException, IOException {
		
		double [][] Phi = {
				{0.70, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05},
				{0.05, 0.70, 0.05, 0.05, 0.05, 0.05, 0.05},
				{0.05, 0.05, 0.70, 0.05, 0.05, 0.05, 0.05},
				{0.05, 0.05, 0.05, 0.70, 0.05, 0.05, 0.05},
				{0.05, 0.05, 0.05, 0.05, 0.70, 0.05, 0.05},
				{0.05, 0.05, 0.05, 0.05, 0.05, 0.70, 0.05},
				{0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.70}
		};

		double [][] etas = {
				{2,    3,    4},
				{-4,   -3,   -2}
		};

		double [][] xs = {
				{1}
		};
		
		int [] zs = {7, 7, 1, 5, 1, 1, 3, 1, 1, 3, 1, 3, 7, 1, 1, 7, 2, 1, 1, 1, 7, 1, 2, 7, 1, 1, 1, 7, 1, 5, 3, 5, 1, 1, 1, 5, 1, 1, 7, 1, 1, 1, 1, 2, 1, 2, 6, 1, 2, 1, 1, 1, 1, 4, 1, 3, 1, 1, 1, 1, 1, 2, 4, 1, 5, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 7, 7, 3, 1, 1, 3, 1};
		
		// Stupid R is 1 indexed
		for (int i = 0; i < zs.length; i++) {
			zs[i] = zs[i]-1;
		}

		int [] ws = {7, 7, 1, 5, 7, 2, 6, 3, 1, 3, 1, 5, 4, 1, 1, 7, 2, 1, 1, 5, 7, 1, 2, 7, 3, 1, 7, 1, 1, 3, 7, 5, 1, 7, 6, 5, 5, 1, 4, 1, 1, 1, 1, 4, 3, 5, 6, 7, 2, 1, 3, 1, 4, 4, 1, 4, 1, 1, 1, 1, 7, 2, 4, 1, 5, 1, 1, 5, 1, 1, 3, 1, 1, 1, 1, 3, 5, 1, 4, 4, 7, 6, 1, 1, 3, 3, 7, 4, 1, 1, 1, 4, 1, 7, 7, 3, 1, 1, 6, 3};

		// Stupid R is 1 indexed
		for (int i = 0; i < ws.length; i++) {
			ws[i] = ws[i]-1;
		}
		
		int Ks = 2;
		double alpha = 0.01;
		double [][] ast = {{4.371304}, {-5.397788}};
		LDADocSamplingContextTest ctx = new LDADocSamplingContextTest(0, Phi, etas, Ks, alpha, xs, ast, ws, Arrays.copyOf(zs, zs.length));
		DOLDAGibbsSamplerMock mock = new DOLDAGibbsSamplerMock();
		mock.sample_z(ctx);		
	}

	@Test
	public void testZGivenBeta1() throws IOException {
		int Ks = 2;
		double [][] as = {{-1, 1, -2}};
		double [][] ast = MatrixOps.transposeSerial(as);
		double [][] xs = {{-1.55, 1.32}};
		double [][] etast = {{7, 2, -8}, {3, 4, -4}, {-3, 0, 0}, {-4, -3, 0}}; 
		double [][] etas  = MatrixOps.transposeSerial(etast); 
		double [][] Phi = {
				{0.15, 0.24, 0.14, 0.11, 0.36},
				{0.39, 0.34, 0.08, 0.04, 0.15},
				{0.10, 0.01, 0.23, 0.35, 0.31},
				{0.24, 0.08, 0.13, 0.14, 0.41}
		};
		int [] ws = {5, 1, 3, 4, 4, 1, 5, 3, 5, 5, 3, 3, 5, 3, 5, 4, 2, 5, 3, 2};
		// Stupid R is 1 indexed
		for (int i = 0; i < ws.length; i++) {
			ws[i] = ws[i]-1;
		}

		int [] zs = {4, 1, 2, 1, 4, 3, 3, 3, 2, 2, 4, 4, 4, 2, 4, 2, 3, 4, 2, 4};
		// Stupid R is 1 indexed
		for (int i = 0; i < zs.length; i++) {
			zs[i] = zs[i]-1;
		}

		String expectedFn = "src/test/resources/dolda/localtopiccounts1.csv";
		DataFrame<Object> df = DataFrame.readCsv(expectedFn, sep, NumberDefault.DOUBLE_DEFAULT, naString, hasHeader);
		DataFrame<Double> ddf = df.cast(Double.class);
		double [][] zsExpect = ddf.transpose().fillna(0.0).toArray(double[][].class);
		double [][] local_zs1 = new double[it][zsExpect.length];

		double alpha = 0.1;

		for(int i = 0; i < it; i++){
			// We need to copy the z's each time, since they are modified by the sampler and in this test we don't want that
			LDADocSamplingContextTest ctx = new LDADocSamplingContextTest(0, Phi, etas, Ks, alpha, xs, ast, ws, Arrays.copyOf(zs, zs.length));
			DOLDAGibbsSamplerMock mock = new DOLDAGibbsSamplerMock();
			int [] tmpzs = mock.sample_z(ctx);
			for (int j = 0; j < tmpzs.length; j++) {				
				local_zs1[i][j] = tmpzs[j];
			}		
		}

		LDAUtils.writeASCIIDoubleMatrix(local_zs1, local_zs1.length, local_zs1[0].length, "javalocaltopiccounts1.csv", sep);
		//System.out.println("Expected: " + MatrixOps.doubleArrayToPrintString(MatrixOps.transposeSerial(zsExpect)));
		//System.out.println("Sampled: " + MatrixOps.doubleArrayToPrintString(local_zsD1));
		
		double[] expectedColMeans = MatrixOps.colMeans(MatrixOps.transposeSerial(zsExpect));
		System.out.println("Expected colMeans: " + MatrixOps.arrToStr(expectedColMeans));
		double[] obsColMeans = MatrixOps.colMeans(local_zs1);
		System.out.println("Observed colMeans: " + MatrixOps.arrToStr(obsColMeans));
		double[] expectedColSd = MatrixOps.colStddev(MatrixOps.transposeSerial(zsExpect));
		System.out.println("Expected expectedColSd: " + MatrixOps.arrToStr(expectedColSd));
		double[] observedColSd = MatrixOps.colStddev(local_zs1);
		System.out.println("Observed observedColSd: " + MatrixOps.arrToStr(observedColSd));
		
		double [][] observedt = MatrixOps.transposeSerial(local_zs1);
		double [] tRes = new double[zsExpect.length];
		TTest tt = new TTest();
		for (int i = 0; i < zsExpect.length; i++) {
			//double p = tt.tTest(zsExpect[i], observedt[i]);
			double p = tt.homoscedasticTTest(zsExpect[i], observedt[i]);
			tRes[i] = p;
			System.out.println("Test " + i + " => " + p + " (" + (p > p_thr) + ")");
		}
		for (int i = 0; i < tRes.length; i++) {			
			assertTrue("T p=" + tRes[i], tRes[i] > p_thr);
		}
	}

	@Test
	public void testZGivenBeta2() throws IOException {
		int Ks = 3;
		//int Ku = 0;
		//int V = 6;
		double [][] as = {{-1, -2, 2, -4}};
		double [][] ast = MatrixOps.transposeSerial(as);
		double [][] xs = {{-7}};
		double [][] etast = {{7, 2, -8, 3}, {4, -4, -3, 0}, {0, -4, -3, 0}, {4,-1,-1,-2}}; 
		double [][] etas  = MatrixOps.transposeSerial(etast); 
		double [][] Phi = {
				{0.02, 0.11, 0.06, 0.22, 0.55, 0.04},
				{0.46, 0.19, 0.15, 0.08, 0.05, 0.06},
				{0.10, 0.01, 0.12, 0.03, 0.16, 0.58}
		};
		
		int [] ws = {5, 5, 3, 2, 5, 4, 1, 2, 3, 5, 4, 2, 2, 3, 5, 2, 6, 6, 6, 3, 6, 3, 6, 2, 5, 3, 1, 3, 4, 4};
		// Stupid R is 1 indexed
		for (int i = 0; i < ws.length; i++) {
			ws[i] = ws[i]-1;
		}

		int [] zs = {3, 3, 2, 1, 2, 2, 2, 2, 1, 3, 2, 3, 3, 3, 2, 1, 1, 2, 2, 2, 3, 1, 1, 1, 1, 3, 2, 2, 2, 3};
		for (int i = 0; i < zs.length; i++) {
			zs[i] = zs[i]-1;
		}
		
		String expectedFn = "src/test/resources/dolda/localtopiccounts2.csv";
		DataFrame<Object> df = DataFrame.readCsv(expectedFn, sep, NumberDefault.DOUBLE_DEFAULT, naString, hasHeader);
		DataFrame<Double> ddf = df.cast(Double.class);
		double [][] zsExpect = ddf.transpose().fillna(0.0).toArray(double[][].class);
		double [][] local_zs1 = new double[it][zsExpect.length];

		double alpha = 10;

		for(int i = 0; i < it; i++){
			// We need to copy the z's each time, since they are modified by the sampler and in this test we don't want that
			LDADocSamplingContextTest ctx = new LDADocSamplingContextTest(0, Phi, etas, Ks, alpha, xs, ast, ws, Arrays.copyOf(zs, zs.length));
			DOLDAGibbsSamplerMock mock = new DOLDAGibbsSamplerMock();
			int [] tmpzs = mock.sample_z(ctx);
			for (int j = 0; j < tmpzs.length; j++) {				
				local_zs1[i][j] = tmpzs[j];
			}
		}
		
		LDAUtils.writeASCIIDoubleMatrix(local_zs1, local_zs1.length, local_zs1[0].length, "javalocaltopiccounts2.csv", sep);
		//System.out.println("Expected: " + MatrixOps.doubleArrayToPrintString(MatrixOps.transposeSerial(zsExpect)));
		//System.out.println("Sampled: " + MatrixOps.doubleArrayToPrintString(local_zsD1));
		
		double[] expectedColMeans = MatrixOps.colMeans(MatrixOps.transposeSerial(zsExpect));
		System.out.println("Expected colMeans: " + MatrixOps.arrToStr(expectedColMeans));
		double[] obsColMeans = MatrixOps.colMeans(local_zs1);
		System.out.println("Observed colMeans: " + MatrixOps.arrToStr(obsColMeans));
		double[] expectedColSd = MatrixOps.colStddev(MatrixOps.transposeSerial(zsExpect));
		System.out.println("Expected expectedColSd: " + MatrixOps.arrToStr(expectedColSd));
		double[] observedColSd = MatrixOps.colStddev(local_zs1);
		System.out.println("Observed observedColSd: " + MatrixOps.arrToStr(observedColSd));
		
		double [][] observedt = MatrixOps.transposeSerial(local_zs1);
		double [] tRes = new double[zsExpect.length];
		TTest tt = new TTest();
		for (int i = 0; i < zsExpect.length; i++) {
			//double p = tt.tTest(zsExpect[i], observedt[i]);
			double p = tt.homoscedasticTTest(zsExpect[i], observedt[i]);
			tRes[i] = p;
			System.out.println("Test " + i + " => " + p + " (" + (p > p_thr) + ")");
		}
		for (int i = 0; i < tRes.length; i++) {			
			assertTrue("T p=" + tRes[i], tRes[i] > p_thr);
		}
	}

	@Test
	public void testZGivenBeta3() throws IOException, InterruptedException {
		int Ks = 2;
		//int Ku = 2;
		//int V = 5;
		double [][] as = {{-2, -0.5, 1}};
		double [][] ast = MatrixOps.transposeSerial(as);
		double [][] xs = {{-1.55, 1.32}};
		double [][] etast = {{1, 1, -1}, {0, 1, -1}, {-1, 0, 0}, {-1, -1, 0}};
		double [][] etas  = MatrixOps.transposeSerial(etast); 
		double [][] Phi = {
				{0.15, 0.24, 0.14, 0.11, 0.36},
				{0.39, 0.34, 0.08, 0.04, 0.15},
				{0.10, 0.01, 0.23, 0.35, 0.31},
				{0.24, 0.08, 0.13, 0.14, 0.41}
		};
		int [] ws = {3, 3, 4, 5, 3, 1, 5, 1, 1, 3, 3, 5, 4, 4, 3, 3, 3, 3, 4, 2, 2, 2, 1, 1, 4, 1, 1, 3, 3, 1, 1, 2, 2, 5, 5};

		// Stupid R is 1 indexed
		for (int i = 0; i < ws.length; i++) {
			ws[i] = ws[i]-1;
		}

		int [] zs = {1, 1, 4, 3, 3, 2, 3, 3, 1, 4, 4, 3, 3, 2, 3, 3, 3, 4, 1, 2, 2, 1, 4, 3, 2, 3, 2, 3, 2, 2, 2, 2, 1, 4, 4};
		// Stupid R is 1 indexed
		for (int i = 0; i < zs.length; i++) {
			zs[i] = zs[i]-1;
		}
		
		String expectedFn = "src/test/resources/dolda/localtopiccounts3.csv";
		DataFrame<Object> df = DataFrame.readCsv(expectedFn, sep, NumberDefault.DOUBLE_DEFAULT, naString, hasHeader);
		DataFrame<Double> ddf = df.cast(Double.class);
		double [][] zsExpect = ddf.transpose().fillna(0.0).toArray(double[][].class);
		double [][] local_zs1 = new double[it][zsExpect.length];

		double alpha = 1;

		for(int i = 0; i < it; i++){
			// We need to copy the z's each time, since they are modified by the sampler and in this test we don't want that
			LDADocSamplingContextTest ctx = new LDADocSamplingContextTest(0, Phi, etas, Ks, alpha, xs, ast, ws, Arrays.copyOf(zs, zs.length));
			DOLDAGibbsSamplerMock mock = new DOLDAGibbsSamplerMock();
			int [] tmpzs = mock.sample_z(ctx);
			for (int j = 0; j < tmpzs.length; j++) {				
				local_zs1[i][j] = tmpzs[j];
			}		
		}
		
		
		LDAUtils.writeASCIIDoubleMatrix(local_zs1, local_zs1.length, local_zs1[0].length, "javalocaltopiccounts3.csv", sep);
		//System.out.println("Expected: " + MatrixOps.doubleArrayToPrintString(MatrixOps.transposeSerial(zsExpect)));
		//System.out.println("Sampled: " + MatrixOps.doubleArrayToPrintString(local_zsD1));
		
		double[] expectedColMeans = MatrixOps.colMeans(MatrixOps.transposeSerial(zsExpect));
		System.out.println("Expected colMeans: " + MatrixOps.arrToStr(expectedColMeans));
		double[] obsColMeans = MatrixOps.colMeans(local_zs1);
		System.out.println("Observed colMeans: " + MatrixOps.arrToStr(obsColMeans));
		double[] expectedColSd = MatrixOps.colStddev(MatrixOps.transposeSerial(zsExpect));
		System.out.println("Expected expectedColSd: " + MatrixOps.arrToStr(expectedColSd));
		double[] observedColSd = MatrixOps.colStddev(local_zs1);
		System.out.println("Observed observedColSd: " + MatrixOps.arrToStr(observedColSd));
		
		double [][] observedt = MatrixOps.transposeSerial(local_zs1);
		double [] tRes = new double[zsExpect.length];
		TTest tt = new TTest();
		for (int i = 0; i < zsExpect.length; i++) {
			//double p = tt.tTest(zsExpect[i], observedt[i]);
			double p = tt.homoscedasticTTest(zsExpect[i], observedt[i]);
			tRes[i] = p;
			System.out.println("Test " + i + " => " + p + " (" + (p > p_thr) + ")");
		}
		for (int i = 0; i < tRes.length; i++) {			
			assertTrue("T p=" + tRes[i], tRes[i] > p_thr);
		}
	}
	
	@Test
	public void testCalcScore() throws IOException {
		double alpha = 1;
		int Ks = 2;
		int Ku = 2;
		//int V = 5;
		double [][] as = {{-2, -0.5, 1}};
		double [][] ast = MatrixOps.transposeSerial(as);
		double [][] xs = {{-1.55, 1.32}};
		double [][] etast = {{1, 1, -1}, {0, 1, -1}, {-1, 0, 0}, {-1, -1, 0}};
		double [][] etas = MatrixOps.transposeSerial(etast);
		double [][] Phi = {
				{0.15, 0.24, 0.14, 0.11, 0.36},
				{0.39, 0.34, 0.08, 0.04, 0.15},
				{0.10, 0.01, 0.23, 0.35, 0.31},
				{0.24, 0.08, 0.13, 0.14, 0.41}
		};
		int [] ws = {3, 3, 4, 5, 3, 1, 5, 1, 1, 3, 3, 5, 4, 4, 3, 3, 3, 3, 4, 2, 2, 2, 1, 1, 4, 1, 1, 3, 3, 1, 1, 2, 2, 5, 5};

		// Stupid R is 1 indexed
		for (int i = 0; i < ws.length; i++) {
			ws[i] = ws[i]-1;
		}

		int [] zs = {1, 1, 4, 3, 3, 2, 3, 3, 1, 4, 4, 3, 3, 2, 3, 3, 3, 4, 1, 2, 2, 1, 4, 3, 2, 3, 2, 3, 2, 2, 2, 2, 1, 4, 4};
		// Stupid R is 1 indexed
		for (int i = 0; i < zs.length; i++) {
			zs[i] = zs[i]-1;
		}
		
		int noTopics = Ks+Ku;
		int[] localTopicCounts = new int[noTopics];

		// Find the non-zero words and topic counts that we have in this document
		for (int position = 0; position < ws.length; position++) {
			int topicInd = zs[position];
			localTopicCounts[topicInd]++;
		}
		
		double [] zbar_not_i = new double[Ks];
		for (int i = 0; i < zbar_not_i.length; i++) {
			zbar_not_i[i] = localTopicCounts[i] / (double) ws.length;
		}
		
		double [] observed_scores = new double[ws.length*Ks];
		int cnt = 0;
		for (int pos = 0; pos < ws.length; pos++) {
			for (int topic = 0; topic < Ks; topic++) {	
				
				double scaling = DOLDAGibbsSampler.calculateSupervisedLogScaling(0, (double) ws.length, zbar_not_i, xs[0], topic, etas, ast);
				scaling = exp(scaling);
				// If scaling goes "overboard" set it to 'a large value'
				if(Double.isInfinite(scaling)) scaling = 10_000;

				double score = (localTopicCounts[topic] + alpha) * Phi[topic][ws[pos]] * scaling;
				
				observed_scores[cnt++] = score;
			}
		}
		
		String expectedFn = "src/test/resources/dolda/supervisedScore.csv";
		DataFrame<Object> df = DataFrame.readCsv(expectedFn, sep, NumberDefault.DOUBLE_DEFAULT, naString, hasHeader);
		DataFrame<Double> ddf = df.cast(Double.class);
		double [] zsExpect = ddf.toArray(double[].class);
		for (int i = 0; i < zsExpect.length; i++) {
			assertEquals("Scores does not match", zsExpect[i], observed_scores[i], 0.000001);
		}
	}
	
	@Test
	public void testCalcScore2() throws IOException {
		//int [] localTopicCounts = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		double alpha = 1;
		int Ks = 2;
		int Ku = 2;
		//int V = 5;
		double [][] as = {{-2, -0.5, 1}};
		double [][] ast = MatrixOps.transposeSerial(as);
		double [][] xs = {{-1.55, 1.32}};
		double [][] etast = {{1, 1, -1}, {0, 1, -1}, {-1, 0, 0}, {-1, -1, 0}};
		double [][] etas = MatrixOps.transposeSerial(etast);
		double [][] Phi = {
				{0.15, 0.24, 0.14, 0.11, 0.36},
				{0.39, 0.34, 0.08, 0.04, 0.15},
				{0.10, 0.01, 0.23, 0.35, 0.31},
				{0.24, 0.08, 0.13, 0.14, 0.41}
		};
		int [] ws = {1, 0};

		int [] zs = {0, 1, 1, 0};
		
		int noTopics = Ks+Ku;
		int[] localTopicCounts = new int[noTopics];

		// Find the non-zero words and topic counts that we have in this document
		for (int position = 0; position < ws.length; position++) {
			int topicInd = zs[position];
			localTopicCounts[topicInd]++;
		}
		
		double [] zbar_not_i = new double[Ks];
		for (int i = 0; i < zbar_not_i.length; i++) {
			zbar_not_i[i] = localTopicCounts[i] / (double) ws.length;
		}
		
		double [] observed_scores = new double[ws.length*Ks];
		int cnt = 0;
		for (int pos = 0; pos < ws.length; pos++) {
			for (int topic = 0; topic < Ks; topic++) {	
				
				double scaling = DOLDAGibbsSampler.calculateSupervisedLogScaling(0, (double) ws.length, zbar_not_i, xs[0], topic, etas, ast);
				System.out.println("Got scaling: " + scaling);
				
				double score = (localTopicCounts[topic] + alpha) * Phi[topic][ws[pos]] * scaling;
				System.out.println("Got score  : " + scaling);
				
				observed_scores[cnt++] = score;
			}
		}
		
		System.out.println("Exp: " + exp(-2.0));
	}
	
	
	
	class DOLDAGibbsSamplerMock {
		public DOLDAGibbsSamplerMock() {

		}
		
		// THIS IS A TEST METHOD ONLY!! It is used to test the internals of sampleTopicAssignmentsParallel 
		// It must be updated if sampleTopicAssignmentsParallel is updated
		int [] sample_z(LDADocSamplingContextTest ctx) {
			int docId = ctx.getDocId();
			double [][] phi = ctx.getPhi();
			double [][] betas = ctx.getBetas();
			int numTopics = phi.length; 
			int ks = ctx.getKs();
			double alpha = ctx.getAlpha();
			double [][] xs = ctx.getX();
			double [][] Ast = ctx.getAst();

			int type, oldTopic, newTopic;

			int [] tokenSequence = ctx.getWs();
			final int docLength = tokenSequence.length;
			double Nd = (double) docLength;

			if(docLength==0) return new int [0];

			//int [] oneDocTopics = Arrays.copyOf(ctx.getZs(), ctx.getZs().length);
			int [] oneDocTopics = ctx.getZs();

			int[] localTopicCounts = new int[numTopics];

			// Find the non-zero words and topic counts that we have in this document
			for (int position = 0; position < docLength; position++) {
				int topicInd = oneDocTopics[position];
				localTopicCounts[topicInd]++;
			}

			//System.out.println("localTopicCounts=" + MatrixOps.arrToStr(localTopicCounts));
			double score, sum;
			double[] topicTermScores = new double[numTopics];
			double [] zbar_not_i = new double[ks];
			for (int i = 0; i < zbar_not_i.length; i++) {
				zbar_not_i[i] = localTopicCounts[i] / Nd;
			}
			//System.out.println("Zbar (init)=" + MatrixOps.arrToStr(zbar_not_i));

			// Additional covariates of doc
			double [] xd = xs[docId];

			//		Iterate over the words in the document
			for (int position = 0; position < docLength; position++) {
				type = tokenSequence[position];
				oldTopic = oneDocTopics[position];
				localTopicCounts[oldTopic]--;
				if(localTopicCounts[oldTopic]<0) 
					throw new IllegalStateException("Counts cannot be negative! Count for topic:" 
							+ oldTopic + " is: " + localTopicCounts[oldTopic]);

				// Propagates the update to the topic-token assignments
				
				// Used to subtract and add 1 to the local structure containing the number of times
				// each token is assigned to a certain topic. Called before and after taking a sample
				// topic assignment z
				 
				//decrement(myBatch,oldTopic,type);
				// Now calculate and add up the scores for each topic for this word
				sum = 0.0;

				// Array of z_bar's (topic means) with topic 'topic's last contribution  removed.
				if(oldTopic < ks) {
					double [] zbtmp = Arrays.copyOf(zbar_not_i, zbar_not_i.length);
					zbar_not_i[oldTopic] = zbar_not_i[oldTopic]-1/Nd;
					if(zbar_not_i[oldTopic]<(0-1e-16)) {
						throw new IllegalStateException("Zbar[" + oldTopic + "] = " + zbar_not_i[oldTopic] +  " is less than zero:" 
								+ MatrixOps.arrToStr(zbar_not_i) + " Nd=" + Nd 
								+  "\nzbar_not_i=" + MatrixOps.arrToStr(zbtmp)
								+ " 1/Nd=" + (1/Nd)
								);
					}
				}

				//double[] concatenated = concatenate(xd,zbar_not_i);

				for (int topic = 0; topic < numTopics; topic++) {
					double scaling = 1.0;
					if(topic < ks) {
						scaling = DOLDAGibbsSampler.calculateSupervisedLogScaling(docId, Nd, zbar_not_i, xd, topic, betas, Ast);
						scaling = exp(scaling);
						// If scaling goes "overboard" set it to 'a large value'
						if(Double.isInfinite(scaling)) scaling = 10_000;
					}
					score = (localTopicCounts[topic] + alpha) * phi[topic][type] * scaling;
					if(score<0.0 || Double.isNaN(score)) { 
						throw new IllegalStateException("Got a broken score: " 
								+ " score=" + score
								+ " topic=" + topic
								+ " localTopicCounts=" + MatrixOps.arrToStr(localTopicCounts) + "\n"
								+ " betas=" + MatrixOps.doubleArrayToPrintString(betas,50,50,10)
								);
					}
					topicTermScores[topic] = score;
					sum += score;
				}
				//System.out.println("topicTermScores=" + MatrixOps.arrToStr(topicTermScores));
				// Choose a random point between 0 and the sum of all topic scores
				// The thread local random performs better in concurrent situations 
				// than the standard random which is thread safe and incurs lock 
				// contention
				double U = ThreadLocalRandom.current().nextDouble();
				double sample = U * sum;

				newTopic = -1;
				while (sample > 0.0) {
					newTopic++;
					sample -= topicTermScores[newTopic];
				} 

				//System.out.println("Sampled: " + newTopic);

				// Make sure we actually sampled a valid topic
				if (newTopic < 0 || newTopic >= numTopics) {
					throw new IllegalStateException("UncollapsedParallelLDA: New sampled topic ( " + newTopic + ") is not valid, valid topics are between 0 and " + (numTopics-1) + ". Score is: " + sum);
				}

				// Put that new topic into the counts
				oneDocTopics[position] = newTopic;
				localTopicCounts[newTopic]++;
				// Add its contribution to z_bar
				if(newTopic < ks) {
					zbar_not_i[newTopic] = zbar_not_i[newTopic]+1/Nd;
				}

				// Propagates the update to the topic-token assignments
				
				// Used to subtract and add 1 to the local structure containing the number of times
				// each token is assigned to a certain topic. Called before and after taking a sample
				// topic assignment z
				 
				//increment(myBatch,newTopic,type);
			}
			return localTopicCounts;
		}
		
	}

}
