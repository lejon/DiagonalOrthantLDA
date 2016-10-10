package xyz.lejon.bayes.models.probit;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.Map;

import org.apache.commons.cli.ParseException;
import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;
import org.junit.Test;

import xyz.lejon.configuration.ConfigFactory;
import xyz.lejon.configuration.DOCommandLineParser;
import xyz.lejon.configuration.DOConfiguration;
import xyz.lejon.eval.EvalResult;
import xyz.lejon.utils.DataSet;
import xyz.lejon.utils.MatrixOps;

public class DoProbitTest {

	@Test
	public void testDoProbitHS() throws IOException, ParseException, ConfigurationException {
		ConfigFactory.resetFactory();
		String [] args = {"--run_cfg=src/main/resources/configuration/DOProbitBasicTest.cfg"};
		DOCommandLineParser cp = new DOCommandLineParser(args);
		DOConfiguration config = (DOConfiguration) ConfigFactory.getMainConfiguration(cp);

		config.forceActivateSubconfig("DO_glass_full");
		DataSet trainingSetData = config.loadTrainingSet();
		DataSet testSetData = config.loadTestSet();

		double [][] xs = trainingSetData.getX();
		int [] ys = trainingSetData.getY();
		Map<String,Integer> labelMap = trainingSetData.getLabelToId();

		DOSampler doProbit  = ModelFactory.get(config, xs, ys, labelMap.size());
		//DOSampler doProbit  = new MultivariateParallelDOSampler(config, xs,ys,labelMap.size());
		doProbit.sample(config.getNoIterations(DOConfiguration.ITERATIONS_DEFAULT));


		double [][] testset;
		int [] testLabels;
		if(testSetData==null) {
			testset = xs;
			testLabels = ys;
		} else {
			testset = testSetData.getX();
			testLabels = testSetData.getY();
		}

		double [][] betas = doProbit.getBetas();
		EvalResult result  = DOEvaluation.evaluate(testset, testLabels, betas);
		double pCorrect = ((double)result.noCorrect)/testset.length * 100;
		System.out.println("% correct: " + pCorrect);
		assertTrue(pCorrect>55);

		double [][] r_probit = {
				{-1.14888902, -0.417530088, -2.04523734, -2.42749910, -7.9276453, -2.61136874},
				{0.15388010,  0.010759962, -1.31250826, -0.14183907, -0.6417745,  1.85503926},
				{-0.14874145, -0.332175982, -0.12855880, -0.37569148,  0.6562017,  0.54325592},
				{1.40701785,  0.104679264,  0.67092354, -0.34453747, -0.4533487, -0.53096356},
				{-0.69340365,  0.048904727, -0.34164389,  0.80001048,  0.9290898,  1.09947273},
				{0.32527212, -0.073230364, -0.77277903, -0.06968267,  0.1790553,  1.12541635},
				{-0.09834037, -0.052704014, -0.62554645,  0.45330992, -6.5465174,  0.07576579},
				{0.11729390,  0.001517696,  0.69666061,  0.42515459, -0.1300550, -1.20815053},
				{0.08005826, -0.153539305, -0.32848784, -0.35690636, -7.0392572,  0.47633415},
				{-0.05613382,  0.078568179, -0.04556598, -0.03174731, -0.9470181, -0.74195236}
		};
		// R version has classes in columns
		r_probit = MatrixOps.transposeSerial(r_probit);

		KolmogorovSmirnovTest ks = new KolmogorovSmirnovTest();
		boolean [] ksResults = new boolean[betas.length];
		for (int i = 0; i < r_probit.length; i++) {
			boolean ksResult = ks.kolmogorovSmirnovTest(betas[i], r_probit[i]) > 0.01;
			ksResults[i] = ksResult;
			if(ksResult) {
				System.out.println("OK!");	
			} else {
				System.out.println("NOK!");
			}
			/*System.out.println("Draws:");
			for (int j = 0; j < r_probit.length; j++) {
				System.out.print(r_probit[j]  + "<=>" + betas[j] + ", ");
			}
			System.out.println();*/
		}
		for (int i = 0; i < ksResults.length; i++) {
			assertTrue("Kolmogorov doesn not like the " + (i+1)  + " class betas", ksResults[i]);
		}
	}

}
