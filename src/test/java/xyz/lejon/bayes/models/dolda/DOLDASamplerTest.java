package xyz.lejon.bayes.models.dolda;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.util.Date;
import java.util.Map;

import org.apache.commons.cli.ParseException;
import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import xyz.lejon.MarkerIFSmokeTest;
import xyz.lejon.bayes.models.probit.DOEvaluation;
import xyz.lejon.bayes.models.probit.DOSampler;
import xyz.lejon.configuration.DOCommandLineParser;
import xyz.lejon.configuration.DOConfiguration;
import xyz.lejon.configuration.DOLDACommandLineParser;
import xyz.lejon.configuration.DOLDAConfiguration;
import xyz.lejon.configuration.ParsedDOConfiguration;
import xyz.lejon.configuration.ParsedDOLDAConfiguration;
import xyz.lejon.eval.EvalResult;
import xyz.lejon.runnables.ExperimentUtils;
import xyz.lejon.runnables.SLDA;
import xyz.lejon.utils.DataSet;
import xyz.lejon.utils.EnhancedConfusionMatrix;
import xyz.lejon.utils.LoggingUtils;
import xyz.lejon.utils.MatrixOps;
import xyz.lejon.utils.Timer;
import cc.mallet.classify.Trial;
import cc.mallet.types.InstanceList;

@Category(MarkerIFSmokeTest.class)
public class DOLDASamplerTest {

	@Test
	public void testDoldaHS() throws IOException, ParseException, ConfigurationException {
		String [] args = {"--run_cfg=src/main/resources/configuration/DOLDABasicTest.cfg"};
		DOLDACommandLineParser cp = new DOLDACommandLineParser(args);
		DOLDAConfiguration config = new ParsedDOLDAConfiguration(cp);
		LoggingUtils lu = new LoggingUtils();
		String expDir = config.getExperimentOutputDirectory("");
		if(!expDir.equals("")) {
			expDir += "/";
		}
		String logSuitePath = "Runs/" + expDir + "RunSuite" + LoggingUtils.getDateStamp();
		System.out.println("Logging to: " + logSuitePath);
		lu.checkAndCreateCurrentLogDir(logSuitePath);

		config.setLoggingUtil(lu);
		config.forceActivateSubconfig("DO_glass");
		
		DOLDADataSet trainingSetData = config.loadCombinedTrainingSet();
		DOLDADataSet testSetData = config.loadCombinedTestSet();
		
		double [][] xs = trainingSetData.getX();
		int [] ys = trainingSetData.getY();

		DOLDA dolda = ModelFactory.get(config, xs, ys);
		
		System.out.println("X is: " + MatrixOps.doubleArrayToPrintString(xs, 5));
		
		int commonSeed = 4711;
		dolda.setRandomSeed(commonSeed);
		System.out.println(String.format("Rare word threshold: %d", config.getRareThreshold(DOLDAConfiguration.RARE_WORD_THRESHOLD)));

		InstanceList textData = trainingSetData.getTextData();
		if(textData!=null) {
			System.out.println("Vocabulary size: " + textData.getDataAlphabet().size() + "\n");
			System.out.println("Instance list is: " + textData.size());
			System.out.println("Loading data instances...");
		}
		// Sets the frequent with which top words for each topic are printed
		//model.setShowTopicsInterval(config.getTopicInterval(DOLDAConfiguration.TOPIC_INTER_DEFAULT));
		dolda.setRandomSeed(config.getSeed(DOLDAConfiguration.SEED_DEFAULT));
		System.out.println("Config seed:" + config.getSeed(DOLDAConfiguration.SEED_DEFAULT));
		System.out.println("Start seed: " + dolda.getStartSeed());
		// Imports the data into the model
		if(textData!=null) {
			dolda.addInstances(textData);
		}
		System.out.println("Starting iterations (" + config.getNoIterations(DOLDAConfiguration.NO_ITER_DEFAULT) + " total).");
		System.out.println("_____________________________\n");

		// Runs the model
		System.out.println("Starting:" + new Date());
		Timer t = new Timer();
		t.start();
		dolda.sample(config.getNoIterations(DOLDAConfiguration.NO_ITER_DEFAULT));
		t.stop();
		System.out.println("Finished:" + new Date());
		
		/* ========= Evaluation stage =============== */ 
		double [][] betas = dolda.getBetas();	
		
		double [][] testset;
		int [] testLabels;
		
		if(testSetData==null) {
			double[][] zbar_d = dolda.getSupervsedTopicIndicatorMeans();
			testset = MatrixOps.concatenate(xs,zbar_d);
			testLabels = ys;
		} else {
			// Here we sample the topic indicators using ordinary LDA 
			// with the Phi we have learned during sampling!
			
			double [][] testZ;
			if(config.getTextDatasetTestFilename()!=null) {
				testZ = SLDA.sampleTestTopicIndicatorsMeans(config, commonSeed, dolda.getPhi());
			} else {
				testZ = new double[testSetData.getX().length][0];
			}
			
			testset = MatrixOps.concatenate(testSetData.getX(),testZ);
			testLabels = testSetData.getY();
		}		EvalResult result  = DOEvaluation.evaluate(testset, testLabels, betas);
		
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
	
	@Category(MarkerIFSmokeTest.class)
	@Test
	public void testPointCrossValidation() throws Exception {
		DOLDAClassifier dcls;
		String [] args = {"--run_cfg=src/main/resources/configuration/DOLDABasicTest.cfg"};
		DOLDACommandLineParser cp = new DOLDACommandLineParser(args);
		DOLDAConfiguration config = new ParsedDOLDAConfiguration(cp);
		LoggingUtils lu = new LoggingUtils();
		String expDir = config.getExperimentOutputDirectory("");
		if(!expDir.equals("")) {
			expDir += "/";
		}
		String logSuitePath = "Runs/" + expDir + "RunSuite" + LoggingUtils.getDateStamp();
		System.out.println("Logging to: " + logSuitePath);
		lu.checkAndCreateCurrentLogDir(logSuitePath);

		config.setLoggingUtil(lu);
		config.forceActivateSubconfig("DOLDA-films-imdb");
		
		DOLDADataSet trainingSetData = config.loadCombinedTrainingSet();

		double [][] xs = trainingSetData.getX();
		
		System.out.println("X is: " + MatrixOps.doubleArrayToPrintString(xs, 5));
		
		System.out.println(String.format("Rare word threshold: %d", config.getRareThreshold(DOLDAConfiguration.RARE_WORD_THRESHOLD)));

		InstanceList textData = trainingSetData.getTextData();
		if(textData!=null) {
			System.out.println("Vocabulary size: " + textData.getDataAlphabet().size() + "\n");
			System.out.println("Instance list is: " + textData.size());
			System.out.println("Loading data instances...");
		}

		dcls = new DOLDAPointClassifier(config, trainingSetData);
		
		int folds = config.getNoXFolds(DOLDAConfiguration.XFOLDS);
		
		Timer t = new Timer();
		t.start();
		Trial [] trials = dcls.crossValidate(textData,folds);
		t.stop();

		String trialResuls = "[";
		double average = 0.0;
		for (int trialNo = 0; trialNo < trials.length; trialNo++) {
			average += trials[trialNo].getAccuracy();
			trialResuls += String.format("%.4f",trials[trialNo].getAccuracy()) + ", ";
		}
		trialResuls = trialResuls.substring(0, trialResuls.length()-1) + "]";
		System.out.println();
		
		DOLDA dolda = ((DOLDAPointClassifier)trials[0].getClassifier()).getSampler();
		double [][] betas = dolda.getBetas();
		System.out.println("Example Betas: " + MatrixOps.doubleArrayToPrintString(betas));
		System.out.println();
		EnhancedConfusionMatrix combinedConfusionMatrix = new EnhancedConfusionMatrix(trials);
		System.out.println("Combined Confusion Matrix: \n" + combinedConfusionMatrix);
		System.out.println();
		double xvalidationAverage = average / trials.length;
		System.out.println("X-validation: " + trialResuls + " average: " + xvalidationAverage);
		System.out.println("DOLDA cross validation took: " + (t.getEllapsedTime() / 1000) + " seconds");

	}
	
	@Category(MarkerIFSmokeTest.class)
	@Test
	public void testDistCrossValidation() throws Exception {
		DOLDAClassifier dcls;
		String [] args = {"--run_cfg=src/main/resources/configuration/DOLDABasicTest.cfg"};
		DOLDACommandLineParser cp = new DOLDACommandLineParser(args);
		DOLDAConfiguration config = new ParsedDOLDAConfiguration(cp);
		LoggingUtils lu = new LoggingUtils();
		String expDir = config.getExperimentOutputDirectory("");
		if(!expDir.equals("")) {
			expDir += "/";
		}
		String logSuitePath = "Runs/" + expDir + "RunSuite" + LoggingUtils.getDateStamp();
		System.out.println("Logging to: " + logSuitePath);
		lu.checkAndCreateCurrentLogDir(logSuitePath);

		config.setLoggingUtil(lu);
		config.forceActivateSubconfig("DOLDA-films-imdb");
		
		DOLDADataSet trainingSetData = config.loadCombinedTrainingSet();

		double [][] xs = trainingSetData.getX();
		
		System.out.println("X is: " + MatrixOps.doubleArrayToPrintString(xs, 5));
		
		System.out.println(String.format("Rare word threshold: %d", config.getRareThreshold(DOLDAConfiguration.RARE_WORD_THRESHOLD)));

		InstanceList textData = trainingSetData.getTextData();
		if(textData!=null) {
			System.out.println("Vocabulary size: " + textData.getDataAlphabet().size() + "\n");
			System.out.println("Instance list is: " + textData.size());
			System.out.println("Loading data instances...");
		}

		dcls = new DOLDASamplingClassifier(config, trainingSetData);
		
		int folds = config.getNoXFolds(DOLDAConfiguration.XFOLDS);
		
		Timer t = new Timer();
		t.start();
		Trial [] trials = dcls.crossValidate(textData,folds);
		t.stop();

		String trialResuls = "[";
		double average = 0.0;
		for (int trialNo = 0; trialNo < trials.length; trialNo++) {
			average += trials[trialNo].getAccuracy();
			trialResuls += String.format("%.4f",trials[trialNo].getAccuracy()) + ", ";
		}
		trialResuls = trialResuls.substring(0, trialResuls.length()-1) + "]";
		System.out.println();
		
		DOLDA dolda = ((DOLDAPointClassifier)trials[0].getClassifier()).getSampler();
		double [][] betas = dolda.getBetas();
		System.out.println("Example Betas: " + MatrixOps.doubleArrayToPrintString(betas));
		System.out.println();
		EnhancedConfusionMatrix combinedConfusionMatrix = new EnhancedConfusionMatrix(trials);
		System.out.println("Combined Confusion Matrix: \n" + combinedConfusionMatrix);
		System.out.println();
		double xvalidationAverage = average / trials.length;
		System.out.println("X-validation: " + trialResuls + " average: " + xvalidationAverage);
		System.out.println("DOLDA cross validation took: " + (t.getEllapsedTime() / 1000) + " seconds");
		
		File lgDir = lu.getLogDir();
		
		String [] xColnames = trainingSetData.getColnamesX();
		if(xColnames==null) { xColnames = new String[0]; }
		String [] allColnames = new String[xColnames.length+config.getNoSupervisedTopics(0)];
		for (int i = 0; i < allColnames.length; i++) {
			if(i<xColnames.length) {
				allColnames[i] = xColnames[i];
			} else {
				allColnames[i] = i + "";
			}
		}

		// Save example betas
		if(config.getSaveBetas()) {
			ExperimentUtils.saveBetas(lgDir, allColnames, xs[0].length, betas, config.getIdMap(), config.betasOutputFn());
		}
		
		// Save example beta samples if that is turned on in config
		if(config.saveBetaSamples()) {
			ExperimentUtils.saveBetaSamples(lgDir, allColnames, xs[0].length, dolda.getSampledBetas(), config.getIdMap(), config.betaSamplesOutputFn());
		}
		
		// Save example doc-topic means if that is turned on in config
		if(config.saveDocumentTopicMeans() && config.getTextDatasetTrainFilename()!=null) {
			ExperimentUtils.saveDocTopicMeans(lgDir, dolda.getZbar(), config.getDocumentTopicMeansOutputFilename());
		}
	}
	
	@Test
	public void testCompareProbits() throws IOException, ParseException, ConfigurationException {
		
		int iterations = 10000;
		double [][] doldaBetas = sampleDOLDAProbit(iterations);
		
		double [][] doprobitBetas = sampleDOProbit(iterations);
		
		System.out.println("DOLDA Betas: " + MatrixOps.doubleArrayToPrintString(doldaBetas));
		System.out.println("DOProbit Betas: " + MatrixOps.doubleArrayToPrintString(doprobitBetas));

		KolmogorovSmirnovTest ks = new KolmogorovSmirnovTest();
		boolean [] ksResults = new boolean[doldaBetas.length];
		for (int i = 0; i < doprobitBetas.length; i++) {
			double p = ks.kolmogorovSmirnovTest(doldaBetas[i], doprobitBetas[i]);
			System.out.println("P is: "+ p);
			boolean ksResult = p > 0.00001;
			ksResults[i] = ksResult;
			if(ksResult) {
				System.out.println("OK!");	
			} else {
				System.out.println("NOK!");
			}
		}
		for (int i = 0; i < ksResults.length; i++) {
			assertTrue("Kolmogorov doesn not like the " + (i+1)  + " class betas", ksResults[i]);
		}
	}
	
	double [][] sampleDOLDAProbit(int iterations) throws ParseException, ConfigurationException, IOException {
		String [] args = {"--normalize","--run_cfg=src/main/resources/configuration/DOLDABasicTest.cfg"};
		DOLDACommandLineParser cp = new DOLDACommandLineParser(args);
		DOLDAConfiguration config = new ParsedDOLDAConfiguration(cp);
		LoggingUtils lu = new LoggingUtils();
		String expDir = config.getExperimentOutputDirectory("");
		if(!expDir.equals("")) {
			expDir += "/";
		}
		String logSuitePath = "Runs/" + expDir + "RunSuite" + LoggingUtils.getDateStamp();
		System.out.println("Logging to: " + logSuitePath);
		lu.checkAndCreateCurrentLogDir(logSuitePath);

		config.setLoggingUtil(lu);
		config.forceActivateSubconfig("DO_glass_full");
		
		DOLDADataSet trainingSetData = config.loadCombinedTrainingSet();
		
		double [][] xs = trainingSetData.getX();
		int [] ys = trainingSetData.getY();

		DOLDA dolda = ModelFactory.get(config, xs, ys);
		
		System.out.println("X is: " + MatrixOps.doubleArrayToPrintString(xs, 5));
		
		int commonSeed = 4711;
		dolda.setRandomSeed(commonSeed);
		System.out.println(String.format("Rare word threshold: %d", config.getRareThreshold(DOLDAConfiguration.RARE_WORD_THRESHOLD)));

		InstanceList textData = trainingSetData.getTextData();
		if(textData!=null) {
			System.out.println("Vocabulary size: " + textData.getDataAlphabet().size() + "\n");
			System.out.println("Instance list is: " + textData.size());
			System.out.println("Loading data instances...");
		}
		// Sets the frequent with which top words for each topic are printed
		//model.setShowTopicsInterval(config.getTopicInterval(DOLDAConfiguration.TOPIC_INTER_DEFAULT));
		dolda.setRandomSeed(config.getSeed(DOLDAConfiguration.SEED_DEFAULT));
		System.out.println("Config seed:" + config.getSeed(DOLDAConfiguration.SEED_DEFAULT));
		System.out.println("Start seed: " + dolda.getStartSeed());
		// Imports the data into the model
		if(textData!=null) {
			dolda.addInstances(textData);
		}
		System.out.println("Starting iterations (" + config.getNoIterations(DOLDAConfiguration.NO_ITER_DEFAULT) + " total).");
		System.out.println("_____________________________\n");

		// Runs the model
		System.out.println("Starting:" + new Date());
		Timer t = new Timer();
		t.start();
		dolda.sample(iterations);
		t.stop();
		System.out.println("Finished:" + new Date());
		 
		return dolda.getBetas();	
	}
	
	double [][] sampleDOProbit(int iterations) throws ParseException, ConfigurationException, IOException {
		String [] args = {"--normalize","--run_cfg=src/main/resources/configuration/DOLDABasicTest.cfg"};
		DOCommandLineParser cp = new DOCommandLineParser(args);
		DOConfiguration config = new ParsedDOConfiguration(cp);
		LoggingUtils lu = new LoggingUtils();
		String logSuitePath = "Runs/RunSuite" + LoggingUtils.getDateStamp();
		System.out.println("Logging to: " + logSuitePath);
		lu.checkAndCreateCurrentLogDir(logSuitePath);

		config.setLoggingUtil(lu);
		config.forceActivateSubconfig("DO_glass_full");
		
		DataSet trainingSetData = config.loadTrainingSet();
		
		double [][] xs = trainingSetData.getX();
		int [] ys = trainingSetData.getY();

		Map<String,Integer> labelMap = trainingSetData.getLabelToId();

		DOSampler doProbit = xyz.lejon.bayes.models.probit.ModelFactory.get(config, xs, ys, labelMap.size());

		System.out.println("X is: " + MatrixOps.doubleArrayToPrintString(xs, 5));
		
		System.out.println("Starting iterations (" + config.getNoIterations(DOLDAConfiguration.NO_ITER_DEFAULT) + " total).");
		System.out.println("_____________________________\n");

		// Runs the model
		System.out.println("Starting:" + new Date());
		Timer t = new Timer();
		t.start();
		doProbit.sample(iterations);
		t.stop();
		System.out.println("Finished:" + new Date());
		 
		return doProbit.getBetas();	
	}

}
