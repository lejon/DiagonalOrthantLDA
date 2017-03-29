package xyz.lejon.configuration;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.List;
import java.util.Map;

import joinery.DataFrame;
import joinery.DataFrame.NumberDefault;
import joinery.impl.Conversion;

import org.apache.commons.cli.ParseException;
import org.apache.commons.configuration.ConfigurationException;
import org.junit.Test;

import xyz.lejon.bayes.models.dolda.DOLDA;
import xyz.lejon.bayes.models.dolda.DOLDADataSet;
import xyz.lejon.bayes.models.dolda.ModelFactory;
import xyz.lejon.utils.LoggingUtils;
import xyz.lejon.utils.MatrixOps;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.util.LDAUtils;
import cc.mallet.util.Randoms;

public class ConfigurationTest {

	@Test
	public void testParsereferenceCategoriesWithQuotes() {
		String cat_ref = "\"DirectedBy\" => \"<none>\", \"ProducedBy\" => \"<none>\", \"Music\" => \"<none>\", \"StoryBy\" => \"<none>\", \"Company\" => \"<none>\"";
		Map<String,String> refCats = ParsedDOConfiguration.parseReferenceCategories(cat_ref);
		assertEquals("<none>", refCats.get("DirectedBy"));
		assertEquals("<none>", refCats.get("ProducedBy"));
		assertEquals("<none>", refCats.get("Music"));
		assertEquals("<none>", refCats.get("StoryBy"));
		assertEquals("<none>", refCats.get("Company"));
	}

	@Test
	public void testParsereferenceCategoriesWithoutQuotes() {
		String cat_ref = "DirectedBy => <none>, ProducedBy => <none>, Music => <none>, StoryBy => <none>, Company => <none>";
		Map<String,String> refCats = ParsedDOConfiguration.parseReferenceCategories(cat_ref);
		assertEquals("<none>", refCats.get("DirectedBy"));
		assertEquals("<none>", refCats.get("ProducedBy"));
		assertEquals("<none>", refCats.get("Music"));
		assertEquals("<none>", refCats.get("StoryBy"));
		assertEquals("<none>", refCats.get("Company"));
	}
	
	@Test
	public void testParsereferenceCategoriesEmpty() {
		String cat_ref = "";
		Map<String,String> refCats = ParsedDOConfiguration.parseReferenceCategories(cat_ref);
		assertEquals(null, refCats);
	}

	@Test
	public void testParsereferenceCategoriesNull() {
		String cat_ref = null;
		Map<String,String> refCats = ParsedDOConfiguration.parseReferenceCategories(cat_ref);
		assertEquals(null, refCats);
	}

	@Test
	public void testParsereferenceCategoriesBrokenEmptyValue() {
		String cat_ref = "DirectedBy => , ProducedBy => <none>, Music => <none>, StoryBy => <none>, Company => <none>";
		try {
			ParsedDOConfiguration.parseReferenceCategories(cat_ref);
			fail("Should have thrown IllegalArgumentException since we have empty value in category");
		} catch (Exception e) {
		}
	}

	@Test
	public void testParsereferenceCategoriesBrokenNoValue() {
		// OBSERVE the difference in the placement of the ','
		String cat_ref = "DirectedBy =>, ProducedBy => <none>, Music => <none>, StoryBy => <none>, Company => <none>";
		try {
			ParsedDOConfiguration.parseReferenceCategories(cat_ref);
			fail("Should have thrown IllegalArgumentException since we have no value in category.");
		} catch (Exception e) {
		}
	}


	@Test
	public void testgetNormalizePropertyTrueInConfigNotInCmdline() throws IOException, ParseException, ConfigurationException {
		ConfigFactory.resetFactory();
		
		String [] args = {"--run_cfg=src/main/resources/configuration/SLDABasicTestConfig.cfg"};
		DOLDACommandLineParser cp = new DOLDACommandLineParser(args);
		DOLDAConfiguration config = (DOLDAConfiguration) ConfigFactory.getMainConfiguration(cp);
		LoggingUtils lu = new LoggingUtils();
		String logSuitePath = "Runs/RunSuite" + LoggingUtils.getDateStamp();
		System.out.println("Logging to: " + logSuitePath);
		lu.checkAndCreateCurrentLogDir(logSuitePath);
		config.setLoggingUtil(lu);
		String conf = "config_with_normailze_true";
		config.forceActivateSubconfig(conf);
		System.out.println("N:" + config.getNormalize());
		assertTrue(config.getNormalize());
	}
	
	@Test
	public void testgetNormalizePropertyFalseInConfigNotInCmdline() throws IOException, ParseException, ConfigurationException {
		ConfigFactory.resetFactory();
		
		String [] args = {"--run_cfg=src/main/resources/configuration/SLDABasicTestConfig.cfg"};
		DOLDACommandLineParser cp = new DOLDACommandLineParser(args);
		DOLDAConfiguration config = (DOLDAConfiguration) ConfigFactory.getMainConfiguration(cp);
		LoggingUtils lu = new LoggingUtils();
		String logSuitePath = "Runs/RunSuite" + LoggingUtils.getDateStamp();
		System.out.println("Logging to: " + logSuitePath);
		lu.checkAndCreateCurrentLogDir(logSuitePath);
		config.setLoggingUtil(lu);
		String conf = "config_with_normailze_false";
		config.forceActivateSubconfig(conf);
		System.out.println("N:" + config.getNormalize());
		assertTrue(!config.getNormalize());
	}
	
	@Test
	public void testgetNormalizePropertyFalseInConfigTrueInCmdline() throws IOException, ParseException, ConfigurationException {
		ConfigFactory.resetFactory();
		
		String [] args = {"--normalize", "--run_cfg=src/main/resources/configuration/SLDABasicTestConfig.cfg"};
		DOLDACommandLineParser cp = new DOLDACommandLineParser(args);
		DOLDAConfiguration config = (DOLDAConfiguration) ConfigFactory.getMainConfiguration(cp);
		LoggingUtils lu = new LoggingUtils();
		String logSuitePath = "Runs/RunSuite" + LoggingUtils.getDateStamp();
		System.out.println("Logging to: " + logSuitePath);
		lu.checkAndCreateCurrentLogDir(logSuitePath);
		config.setLoggingUtil(lu);
		String conf = "config_with_normailze_false";
		config.forceActivateSubconfig(conf);
		System.out.println("N:" + config.getNormalize());
		assertTrue(config.getNormalize());
	}
	
	@Test
	public void testgetNormalizePropertyNotInConfigNotInCmdline() throws IOException, ParseException, ConfigurationException {
		ConfigFactory.resetFactory();
		
		String [] args = {"--run_cfg=src/main/resources/configuration/SLDABasicTestConfig.cfg"};
		DOLDACommandLineParser cp = new DOLDACommandLineParser(args);
		DOLDAConfiguration config = (DOLDAConfiguration) ConfigFactory.getMainConfiguration(cp);
		LoggingUtils lu = new LoggingUtils();
		String logSuitePath = "Runs/RunSuite" + LoggingUtils.getDateStamp();
		System.out.println("Logging to: " + logSuitePath);
		lu.checkAndCreateCurrentLogDir(logSuitePath);
		config.setLoggingUtil(lu);
		String conf = "config_without_normailze";
		config.forceActivateSubconfig(conf);
		System.out.println("N:" + config.getNormalize());
		assertTrue(!config.getNormalize());
	}
	
	@SuppressWarnings("unused")
	@Test
	public void testZOnly() throws IOException, ParseException, ConfigurationException {
		// Needed for testing only....
		ConfigFactory.resetFactory();
		
		String [] args = {"--run_cfg=src/main/resources/configuration/SLDABasicTestConfig.cfg"};
		DOLDACommandLineParser cp = new DOLDACommandLineParser(args);

		DOLDAConfiguration config = (DOLDAConfiguration) ConfigFactory.getMainConfiguration(cp);
		LoggingUtils lu = new LoggingUtils();
		String expDir = config.getExperimentOutputDirectory("");
		if(!expDir.equals("")) {
			expDir += "/";
		}
		String logSuitePath = "Runs/" + expDir + "RunSuite" + LoggingUtils.getDateStamp();
		System.out.println("Logging to: " + logSuitePath);
		lu.checkAndCreateCurrentLogDir(logSuitePath);
		config.setLoggingUtil(lu);

		int commonSeed = config.getSeed(DOLDAConfiguration.SEED_DEFAULT);
		String conf = "Z-only";
		lu.checkCreateAndSetSubLogDir(conf);
		config.activateSubconfig(conf);

		String dataset_fn = config.getTextDatasetTrainFilename();

		DOLDADataSet trainingSetData =config.loadCombinedTrainingSet();
		DOLDADataSet testSetData = config.loadCombinedTestSet();

		double [][] xs = trainingSetData.getX();
		int [] ys = trainingSetData.getY();
		String [] labels = config.getLabels();
		Map<String,Integer> labelMap = config.getLabelMap();
		Map<Integer,String> idMap = config.getIdMap();

		DOLDA dolda = ModelFactory.get(config, xs, ys);

		System.out.println("X is: " + MatrixOps.doubleArrayToPrintString(xs, 5));
		assertEquals(xs.length,trainingSetData.getTextData().size());
		assertEquals(xs[0].length,0);

		dolda.setRandomSeed(commonSeed);
		System.out.println(String.format("Rare word threshold: %d", config.getRareThreshold(DOLDAConfiguration.RARE_WORD_THRESHOLD)));

		System.out.println("Vocabulary size: " + trainingSetData.getTextData().getDataAlphabet().size() + "\n");
		System.out.println("Instance list is: " + trainingSetData.getTextData().size());
		System.out.println("Loading data instances...");

		// Sets the frequent with which top words for each topic are printed
		//model.setShowTopicsInterval(config.getTopicInterval(DOLDAConfiguration.TOPIC_INTER_DEFAULT));
		dolda.setRandomSeed(config.getSeed(DOLDAConfiguration.SEED_DEFAULT));
		System.out.println("Config seed:" + config.getSeed(DOLDAConfiguration.SEED_DEFAULT));
		System.out.println("Start seed: " + dolda.getStartSeed());
		// Imports the data into the model
		dolda.addInstances(trainingSetData.getTextData());

		System.out.println("Starting iterations (" + config.getNoIterations(DOLDAConfiguration.NO_ITER_DEFAULT) + " total).");
		System.out.println("_____________________________\n");

		dolda.sample(config.getNoIterations(DOLDAConfiguration.NO_ITER_DEFAULT));
		
		double [][] betas = dolda.getBetas();
		assertEquals(15,betas.length);
		assertEquals(5,betas[0].length);
		assertEquals((int)config.getNoSupervisedTopics(-1), 5);
		assertEquals((int)config.getNoSupervisedTopics(-1), betas[0].length);
	}
	
	@SuppressWarnings("unused")
	@Test
	public void testZOnlyWithIntercept() throws IOException, ParseException, ConfigurationException {
		// Needed for testing only....
		ConfigFactory.resetFactory();
		
		String [] args = {"--run_cfg=src/main/resources/configuration/SLDABasicTestConfig.cfg"};
		DOLDACommandLineParser cp = new DOLDACommandLineParser(args);

		DOLDAConfiguration config = (DOLDAConfiguration) ConfigFactory.getMainConfiguration(cp);
		LoggingUtils lu = new LoggingUtils();
		String expDir = config.getExperimentOutputDirectory("");
		if(!expDir.equals("")) {
			expDir += "/";
		}
		String logSuitePath = "Runs/" + expDir + "RunSuite" + LoggingUtils.getDateStamp();
		System.out.println("Logging to: " + logSuitePath);
		lu.checkAndCreateCurrentLogDir(logSuitePath);
		config.setLoggingUtil(lu);

		int commonSeed = config.getSeed(DOLDAConfiguration.SEED_DEFAULT);
		String conf = "Z-only-with-intercept";
		lu.checkCreateAndSetSubLogDir(conf);
		config.forceActivateSubconfig(conf);

		String dataset_fn = config.getTextDatasetTrainFilename();

		DOLDADataSet trainingSetData =config.loadCombinedTrainingSet();
		DOLDADataSet testSetData = config.loadCombinedTestSet();

		double [][] xs = trainingSetData.getX();
		int [] ys = trainingSetData.getY();
		String [] labels = config.getLabels();
		Map<String,Integer> labelMap = config.getLabelMap();
		Map<Integer,String> idMap = config.getIdMap();

		DOLDA dolda = ModelFactory.get(config, xs, ys);

		System.out.println("X is: " + MatrixOps.doubleArrayToPrintString(xs, 5));
		assertEquals(xs.length,trainingSetData.getTextData().size());
		assertEquals(xs[0].length,1);

		dolda.setRandomSeed(commonSeed);
		System.out.println(String.format("Rare word threshold: %d", config.getRareThreshold(DOLDAConfiguration.RARE_WORD_THRESHOLD)));

		System.out.println("Vocabulary size: " + trainingSetData.getTextData().getDataAlphabet().size() + "\n");
		System.out.println("Instance list is: " + trainingSetData.getTextData().size());
		System.out.println("Loading data instances...");

		// Sets the frequent with which top words for each topic are printed
		//model.setShowTopicsInterval(config.getTopicInterval(DOLDAConfiguration.TOPIC_INTER_DEFAULT));
		dolda.setRandomSeed(config.getSeed(DOLDAConfiguration.SEED_DEFAULT));
		System.out.println("Config seed:" + config.getSeed(DOLDAConfiguration.SEED_DEFAULT));
		System.out.println("Start seed: " + dolda.getStartSeed());
		// Imports the data into the model
		dolda.addInstances(trainingSetData.getTextData());

		System.out.println("Starting iterations (" + config.getNoIterations(DOLDAConfiguration.NO_ITER_DEFAULT) + " total).");
		System.out.println("_____________________________\n");

		dolda.sample(config.getNoIterations(DOLDAConfiguration.NO_ITER_DEFAULT));
		
		double [][] betas = dolda.getBetas();
		assertEquals(15,betas.length);
		assertEquals(6,betas[0].length);
		assertEquals((int)config.getNoSupervisedTopics(-1), 5);
		assertEquals((int)config.getNoSupervisedTopics(-1), betas[0].length-1);
	}
	
	@SuppressWarnings("unused")
	@Test
	public void testXOnly() throws IOException, ParseException, ConfigurationException {
		// Needed for testing only....
		ConfigFactory.resetFactory();
		
		String [] args = {"--run_cfg=src/main/resources/configuration/SLDABasicTestConfig.cfg"};
		DOLDACommandLineParser cp = new DOLDACommandLineParser(args);

		DOLDAConfiguration config = (DOLDAConfiguration) ConfigFactory.getMainConfiguration(cp);
		LoggingUtils lu = new LoggingUtils();
		String expDir = config.getExperimentOutputDirectory("");
		if(!expDir.equals("")) {
			expDir += "/";
		}
		String logSuitePath = "Runs/" + expDir + "RunSuite" + LoggingUtils.getDateStamp();
		System.out.println("Logging to: " + logSuitePath);
		lu.checkAndCreateCurrentLogDir(logSuitePath);
		config.setLoggingUtil(lu);

		int commonSeed = config.getSeed(DOLDAConfiguration.SEED_DEFAULT);
		String conf = "X-only";
		lu.checkCreateAndSetSubLogDir(conf);
		config.forceActivateSubconfig(conf);

		String dataset_fn = config.getTextDatasetTrainFilename();

		DOLDADataSet trainingSetData =config.loadCombinedTrainingSet();
		DOLDADataSet testSetData = config.loadCombinedTestSet();
		
		assertEquals("Intercept", trainingSetData.getColnamesX()[0]);

		double [][] xs = trainingSetData.getX();
		int [] ys = trainingSetData.getY();
		String [] labels = config.getLabels();
		Map<String,Integer> labelMap = config.getLabelMap();
		Map<Integer,String> idMap = config.getIdMap();

		DOLDA dolda = ModelFactory.get(config, xs, ys);

		System.out.println("X is: " + MatrixOps.doubleArrayToPrintString(xs, 5));
		//assertEquals(null,trainingSetData.getTextData());
		assertEquals(416, xs[0].length);

		dolda.setRandomSeed(commonSeed);

		// Sets the frequent with which top words for each topic are printed
		//model.setShowTopicsInterval(config.getTopicInterval(DOLDAConfiguration.TOPIC_INTER_DEFAULT));
		dolda.setRandomSeed(config.getSeed(DOLDAConfiguration.SEED_DEFAULT));
		System.out.println("Config seed:" + config.getSeed(DOLDAConfiguration.SEED_DEFAULT));
		System.out.println("Start seed: " + dolda.getStartSeed());
		// Imports the data into the model
		//dolda.addInstances(trainingSetData.getTextData());

		System.out.println("Starting iterations (" + config.getNoIterations(DOLDAConfiguration.NO_ITER_DEFAULT) + " total).");
		System.out.println("_____________________________\n");

		dolda.sample(config.getNoIterations(DOLDAConfiguration.NO_ITER_DEFAULT));
		
		double [][] betas = dolda.getBetas();

		System.out.println(MatrixOps.doubleArrayToPrintString(betas,5,5,10));
		assertTrue(betas[0].length==(xs[0].length));
	}
	
	//X-only-no-intercept
	@SuppressWarnings("unused")
	@Test
	public void testXOnlyNoIntercept() throws IOException, ParseException, ConfigurationException {
		// Needed for testing only....
		ConfigFactory.resetFactory();
		
		String [] args = {"--run_cfg=src/main/resources/configuration/SLDABasicTestConfig.cfg"};
		DOLDACommandLineParser cp = new DOLDACommandLineParser(args);

		DOLDAConfiguration config = (DOLDAConfiguration) ConfigFactory.getMainConfiguration(cp);
		LoggingUtils lu = new LoggingUtils();
		String expDir = config.getExperimentOutputDirectory("");
		if(!expDir.equals("")) {
			expDir += "/";
		}
		String logSuitePath = "Runs/" + expDir + "RunSuite" + LoggingUtils.getDateStamp();
		System.out.println("Logging to: " + logSuitePath);
		lu.checkAndCreateCurrentLogDir(logSuitePath);
		config.setLoggingUtil(lu);

		int commonSeed = config.getSeed(DOLDAConfiguration.SEED_DEFAULT);
		String conf = "X-only-no-intercept";
		lu.checkCreateAndSetSubLogDir(conf);
		config.forceActivateSubconfig(conf);

		String dataset_fn = config.getTextDatasetTrainFilename();

		DOLDADataSet trainingSetData =config.loadCombinedTrainingSet();
		DOLDADataSet testSetData = config.loadCombinedTestSet();
		
		assertTrue(!"Intercept".equals(trainingSetData.getColnamesX()[0]));

		double [][] xs = trainingSetData.getX();
		int [] ys = trainingSetData.getY();
		String [] labels = config.getLabels();
		Map<String,Integer> labelMap = config.getLabelMap();
		Map<Integer,String> idMap = config.getIdMap();

		DOLDA dolda = ModelFactory.get(config, xs, ys);

		System.out.println("X is: " + MatrixOps.doubleArrayToPrintString(xs, 5));
		//assertEquals(null,trainingSetData.getTextData());
		assertEquals(415, xs[0].length);

		dolda.setRandomSeed(commonSeed);

		// Sets the frequent with which top words for each topic are printed
		//model.setShowTopicsInterval(config.getTopicInterval(DOLDAConfiguration.TOPIC_INTER_DEFAULT));
		dolda.setRandomSeed(config.getSeed(DOLDAConfiguration.SEED_DEFAULT));
		System.out.println("Config seed:" + config.getSeed(DOLDAConfiguration.SEED_DEFAULT));
		System.out.println("Start seed: " + dolda.getStartSeed());
		// Imports the data into the model
		//dolda.addInstances(trainingSetData.getTextData());

		System.out.println("Starting iterations (" + config.getNoIterations(DOLDAConfiguration.NO_ITER_DEFAULT) + " total).");
		System.out.println("_____________________________\n");

		dolda.sample(config.getNoIterations(DOLDAConfiguration.NO_ITER_DEFAULT));
		
		double [][] betas = dolda.getBetas();

		System.out.println(MatrixOps.doubleArrayToPrintString(betas,5,5,10));
		assertTrue(betas[0].length==(xs[0].length));
	}
	
	@Test
	public void testLoadDatasetColNo() throws IOException, ParseException, ConfigurationException {
		String [] expectedIds = {
				"movie-0",
				"movie-1",
				"movie-2",
				"movie-3",
				"movie-4"};
		DataFrame<Object> df = DataFrame.readCsv("src/test/resources/datasets/small.csv");
		String [] ids = ParsedDOConfiguration.extractIds(df, 0, null);
		assertArrayEquals( expectedIds, ids );
	}

	@Test
	public void testLoadDatasetColName() throws IOException, ParseException, ConfigurationException {
		String [] expectedIds = {
				"movie-0",
				"movie-1",
				"movie-2",
				"movie-3",
				"movie-4"};
		DataFrame<Object> df = DataFrame.readCsv("src/test/resources/datasets/small.csv");
		String [] ids = ParsedDOConfiguration.extractIds(df, -1, "Id");
		assertArrayEquals( expectedIds, ids );
	}
	
	@Test
	public void testLoadDatasetBoth() throws IOException, ParseException, ConfigurationException {
		String [] expectedIds = {
				"movie-0",
				"movie-1",
				"movie-2",
				"movie-3",
				"movie-4"};
		DataFrame<Object> df = DataFrame.readCsv("src/test/resources/datasets/small.csv");
		String [] ids = ParsedDOConfiguration.extractIds(df, 5, "Id");
		assertArrayEquals( expectedIds, ids );
	}
	
	@Test
	public void testLoadLDAIds() throws IOException, ParseException, ConfigurationException {
		String [] expectedIds = {
				"movie-0",
				"movie-1",
				"movie-2",
				"movie-3",
				"movie-4"};
		InstanceList ti = LDAUtils.loadInstances("src/test/resources/datasets/small.lda", null, 1);
		List<String> idList = DOLDAConfigUtils.extractLDAIds(ti);
		String [] ids = new String[idList.size()];
		idList.toArray(ids);
		assertArrayEquals( expectedIds, ids );
	}
	
	@Test
	public void testLoadLDALabels() throws IOException, ParseException, ConfigurationException {
		String [] expectedIds = {
				"Drama",
				"Action",
				"Action",
				"Action",
				"Comedy"};
		InstanceList ti = LDAUtils.loadInstances("src/test/resources/datasets/small.lda", null, 1);
		List<String> idList = DOLDAConfigUtils.extractLDALabels(ti);
		String [] ids = new String[idList.size()];
		idList.toArray(ids);
		assertArrayEquals( expectedIds, ids );
	}
	
	@Test
	public void testExtractXsDeterministic() throws IOException {
		String [] allIds = {
				"movie-0",
				"movie-1",
				"movie-2",
				"movie-3",
				"movie-4"};
		
		// Load textdata
		InstanceList ti = LDAUtils.loadInstances("src/test/resources/datasets/small.lda", null, 1);
		List<String> idList = DOLDAConfigUtils.extractLDAIds(ti);
		String [] ids = new String[idList.size()];
		idList.toArray(ids);
		assertArrayEquals( allIds, ids );

		// Load additional covariates
		DataFrame<Object> df = DataFrame.readCsv("src/test/resources/datasets/small.csv");
		String [] Xids = ParsedDOConfiguration.extractIds(df, 5, "Id");
		
		//System.out.println("Xids = " + Arrays.toString(Xids));
		
		assertArrayEquals( allIds, Xids );
		
		DataFrame<Number> mmdf = Conversion.toModelMatrixDataFrame(df,null,false, null, null);

		//System.out.println(mmdf.columns());
		double [][] xs = mmdf.fillna(0.0).toArray(double[][].class);
		//System.out.println("XS = " + MatrixOps.doubleArrayToPrintString(xs));
		
		String [] expectedIdsTrain = {
				"movie-0",
				"movie-1"};
		
		int TRAINING = 0;
		int TEST = 1;
		InstanceList[] instanceLists = ti.splitInOrder(new double [] {0.5, 0.5});

		int cnt = 0;
		for (Instance instance : instanceLists[TRAINING]) {
			//System.out.println("Training contains: " + instance.getName().toString());
			assertEquals(expectedIdsTrain[cnt++], instance.getName().toString());
		}

		double [][] extracted = DOLDAConfigUtils.extractXs(instanceLists[TRAINING], xs, Xids);
		
		//System.out.println("extracted = " + MatrixOps.doubleArrayToPrintString(extracted));
		assertEquals(expectedIdsTrain.length, extracted.length);
		assertEquals(2014, extracted[0][8],0.00001);
		assertEquals(2014, extracted[1][8],0.00001);

		String [] expectedIdsTest = {
				"movie-2",
				"movie-3",
				"movie-4"};
		
		cnt = 0;
		for (Instance instance : instanceLists[TEST]) {
			//System.out.println("Test contains: " + instance.getName().toString());
			assertEquals(expectedIdsTest[cnt++], instance.getName().toString());
		}
		
		extracted = DOLDAConfigUtils.extractXs(instanceLists[TEST], xs, Xids);
		
		assertEquals(expectedIdsTest.length, extracted.length);
		assertEquals(1983, extracted[0][8], 0.000001);
		assertEquals(1987, extracted[1][8], 0.000001);
		assertEquals(1982, extracted[2][8], 0.000001);
	}

	@Test
	public void testExtractXsRandom() throws IOException {
		String [] allIds = {
				"movie-0",
				"movie-1",
				"movie-2",
				"movie-3",
				"movie-4"};
		
		// Load textdata
		InstanceList ti = LDAUtils.loadInstances("src/test/resources/datasets/small.lda", null, 1);
		List<String> idList = DOLDAConfigUtils.extractLDAIds(ti);
		String [] ids = new String[idList.size()];
		idList.toArray(ids);
		assertArrayEquals( allIds, ids );

		// Load additional covariates
		DataFrame<Object> df = DataFrame.readCsv("src/test/resources/datasets/small.csv");
		String [] Xids = ParsedDOConfiguration.extractIds(df, 5, "Id");
		
		//System.out.println("Xids = " + Arrays.toString(Xids));
		
		assertArrayEquals( allIds, Xids );
		
		DataFrame<Number> mmdf = Conversion.toModelMatrixDataFrame(df,null,false, null, null);

		//System.out.println(mmdf.columns());
		double [][] xs = mmdf.fillna(0.0).toArray(double[][].class);
		//System.out.println("XS = " + MatrixOps.doubleArrayToPrintString(xs));
		
		InstanceList[] instanceLists =
				ti.split(new Randoms(),
						new double[] {0.6, 0.4, 0.0});
				
		int TRAINING = 0;
		int TEST = 1;

		double [][] extracted = DOLDAConfigUtils.extractXs(instanceLists[TRAINING], xs, Xids);
		System.out.println(MatrixOps.doubleArrayToPrintString(extracted));

		assertEquals(instanceLists[TRAINING].size(), extracted.length);
		
		int extractedIdx = 0;
		for (Instance instance : instanceLists[TRAINING]) {
			String instanceName = instance.getName().toString();
			for (int i = 0; i < df.length(); i++) {
				if(df.get(i, 0).equals(instanceName)) {
					assertEquals(((Long) df.get(i, 2)).longValue(), extracted[extractedIdx++][8], 0.0001);
				}
			}
		}

		extracted = DOLDAConfigUtils.extractXs(instanceLists[TEST], xs, Xids);

		assertEquals(instanceLists[TEST].size(), extracted.length);
		extractedIdx = 0;
		for (Instance instance : instanceLists[TEST]) {
			String instanceName = instance.getName().toString();
			for (int i = 0; i < df.length(); i++) {
				if(df.get(i, 0).equals(instanceName)) {
					assertEquals(((Long) df.get(i, 2)).longValue(), extracted[extractedIdx++][8], 0.0001);
				}
			}
		}
	}

	@Test
	public void parseNumericOnlyIds() throws IOException {
		String csvFile = "Id, Name\n 1, apa\n 2, katt";
		InputStream is = new ByteArrayInputStream( csvFile.getBytes( Charset.defaultCharset() ) );
		DataFrame<Object> df = DataFrame.readCsv(is, ",", NumberDefault.LONG_DEFAULT);
		System.out.println(df);
		ParsedDOConfiguration.extractIds(df, 0, "Id");
	}

}
