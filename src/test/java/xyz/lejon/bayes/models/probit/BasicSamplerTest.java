package xyz.lejon.bayes.models.probit;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.util.Map;

import org.apache.commons.cli.ParseException;
import org.apache.commons.configuration.ConfigurationException;
import org.junit.Test;

import xyz.lejon.configuration.ConfigFactory;
import xyz.lejon.configuration.DOCommandLineParser;
import xyz.lejon.configuration.DOConfiguration;
import xyz.lejon.eval.EvalResult;
import xyz.lejon.utils.DataSet;

public class BasicSamplerTest {

	@Test
	public void testNonMatchingTrainTest() throws ParseException, ConfigurationException, IOException {
		ConfigFactory.resetFactory();
		String [] args = {"--run_cfg=src/main/resources/configuration/DOProbitBasicTest.cfg"};
		DOCommandLineParser cp = new DOCommandLineParser(args);
		DOConfiguration config = (DOConfiguration) ConfigFactory.getMainConfiguration(cp);

		String conf = "films-non_matching-labels";
		config.forceActivateSubconfig(conf);
		
		try {
			config.loadTrainingSet();
			fail("Should have thrown IllegalArgumentException since we have non-matching test and train datasets");
		} catch (IllegalArgumentException e) {
		}
	}

	@Test
	public void testSmokeSerial() throws ParseException, ConfigurationException, IOException {
		ConfigFactory.resetFactory();
		String [] args = {"--run_cfg=src/main/resources/configuration/DOProbitBasicTest.cfg"};
		DOCommandLineParser cp = new DOCommandLineParser(args);
		DOConfiguration config = (DOConfiguration) ConfigFactory.getMainConfiguration(cp);

		String [] configs = config.getSubConfigs();
		for(String conf : configs) {
			config.activateSubconfig(conf);
			DataSet trainingSetData = config.loadTrainingSet();
			DataSet testSetData = config.loadTestSet();

			double [][] xs = trainingSetData.getX();
			int [] ys = trainingSetData.getY();
			Map<String,Integer> labelMap = trainingSetData.getLabelToId();

			DOSampler doProbit  = new SerialDOSampler(config, xs,ys,labelMap.size());
			doProbit.sample(config.getNoIterations(DOConfiguration.ITERATIONS_DEFAULT));

			double [][] betas = doProbit.getBetas();

			double [][] testset;
			int [] testLabels;
			if(testSetData==null) {
				testset = xs;
				testLabels = ys;
			} else {
				testset = testSetData.getX();
				testLabels = testSetData.getY();
			}

			EvalResult result  = DOEvaluation.evaluate(testset, testLabels, betas);
			double pCorrect = ((double)result.noCorrect)/testset.length * 100;
			assertTrue(pCorrect>0.75);
		}
	}

	/*
	@Test
	public void testSmokeMultivariate() throws ParseException, ConfigurationException, IOException {
		String [] args = {"--run_cfg=src/main/resources/configuration/DOProbitBasicTest.cfg"};
		DOCommandLineParser cp = new DOCommandLineParser(args);
		DOConfiguration config = (DOConfiguration) ConfigFactory.getMainConfiguration(cp);

		String [] configs = config.getSubConfigs();
		for(String conf : configs) {
			config.activateSubconfig(conf);
			DataSet trainingSetData = config.loadTrainingSet();
			DataSet testSetData = config.loadTestSet();

			double [][] xs = trainingSetData.getX();
			int [] ys = trainingSetData.getY();
			Map<String,Integer> labelMap = trainingSetData.getLabelToId();

			DOSampler doProbit  = new MultivariateParallelDOSampler(config, xs,ys,labelMap.size());
			doProbit.sample(config.getNoIterations(DOConfiguration.ITERATIONS_DEFAULT));

			double [][] betas = doProbit.getBetas();

			double [][] testset;
			int [] testLabels;
			if(testSetData==null) {
				testset = xs;
				testLabels = ys;
			} else {
				testset = testSetData.getX();
				testLabels = testSetData.getY();
			}

			EvalResult result  = DOEvaluation.evaluate(testset, testLabels, betas);
			double pCorrect = ((double)result.noCorrect)/testset.length * 100;
			assertTrue(pCorrect>0.75);
		}
	}

	@Test
	public void testSmokeParallel() throws ParseException, ConfigurationException, IOException {
		String [] args = {"--run_cfg=src/main/resources/configuration/DOProbitBasicTest.cfg"};
		DOCommandLineParser cp = new DOCommandLineParser(args);
		DOConfiguration config = (DOConfiguration) ConfigFactory.getMainConfiguration(cp);

		String [] configs = config.getSubConfigs();
		for(String conf : configs) {
			config.activateSubconfig(conf);
			DataSet trainingSetData = config.loadTrainingSet();
			DataSet testSetData = config.loadTestSet();

			double [][] xs = trainingSetData.getX();
			int [] ys = trainingSetData.getY();
			Map<String,Integer> labelMap = trainingSetData.getLabelToId();

			DOSampler doProbit  = new ParallelDOSampler(config, xs,ys,labelMap.size());
			doProbit.sample(config.getNoIterations(DOConfiguration.ITERATIONS_DEFAULT));

			double [][] betas = doProbit.getBetas();

			double [][] testset;
			int [] testLabels;
			if(testSetData==null) {
				testset = xs;
				testLabels = ys;
			} else {
				testset = testSetData.getX();
				testLabels = testSetData.getY();
			}

			EvalResult result  = DOEvaluation.evaluate(testset, testLabels, betas);
			double pCorrect = ((double)result.noCorrect)/testset.length * 100;
			assertTrue(pCorrect>0.75);
		}
	}*/

}
