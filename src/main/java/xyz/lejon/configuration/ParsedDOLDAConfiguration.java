package xyz.lejon.configuration;

import java.io.IOException;
import java.lang.Thread.UncaughtExceptionHandler;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.configuration.ConfigurationException;

import cc.mallet.types.Alphabet;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelAlphabet;
import cc.mallet.util.LDAUtils;
import xyz.lejon.bayes.models.dolda.DOLDADataSet;
import xyz.lejon.utils.DataSet;
import xyz.lejon.utils.EmptyInstanceIterator;
import xyz.lejon.utils.LoggingUtils;
import xyz.lejon.utils.MatrixOps;


public class ParsedDOLDAConfiguration extends ParsedDOConfiguration implements DOLDAConfiguration, Configuration {

	private static final long serialVersionUID = 1L;
	private int noTopics = -1;
	private int noSupervisedTopics = -1;
	ConfigurationUtils cu = new ConfigurationUtils();
	InstanceList trainingInstances = null;
	InstanceList testInstances;
	
	LoggingUtils logger; 
	
	UncaughtExceptionHandler eh = null;
	
	public ParsedDOLDAConfiguration(DOLDACommandLineParser cp) throws ConfigurationException {
		super(cp.getConfigFn());
		whereAmI = cp.getConfigFn();
		parsedCommandLine = cp.getParsedCommandLine();
		
		if( parsedCommandLine.hasOption( "cm" ) ) {
			comment = parsedCommandLine.getOptionValue( "comment" );
		}
		if( parsedCommandLine.hasOption( "cf" ) ) {
			configFn = parsedCommandLine.getOptionValue( "run_cfg" );
		}

		if (parsedCommandLine.hasOption( "iterations" )) {
			iterations = Integer.parseInt(parsedCommandLine.getOptionValue("iterations").trim());
		}

		if (parsedCommandLine.hasOption( "no_headers" )) {
			hasHeader = false;
		}

		if (parsedCommandLine.hasOption( "na_string" )) {
			naString = parsedCommandLine.getOptionValue("na_string").trim();
		}

		if (parsedCommandLine.hasOption( "plot" )) {
			doPlot = true;
		}

		if (parsedCommandLine.hasOption( "betas_output_file" )) {
			doSave = true;
		}
		
		if (parsedCommandLine.hasOption( "intercept" )) {
			int icept = Integer.parseInt(parsedCommandLine.getOptionValue("intercept").trim());
			if(icept!=0 && icept!=1 ) 
				throw new IllegalArgumentException("Intercept must be 0 (false) or 1 (true),  " + icept + " is not a legal value");
			addIntercept = icept == 1;
			if(addIntercept) System.out.println("Using intercept...");
		}

		sep = parsedCommandLine.hasOption( "separator" ) ? (String) parsedCommandLine.getOptionValue("separator").trim() : null;

		if(sep != null && (!(sep.equals(",") || sep.equals(";") || sep.equals("\\t")))) {
			System.out.println("Only the separators ',' , ';' or '\\t' is currently supported...");
			System.exit(255);
		}
	}

	public void setLoggingUtil(LoggingUtils logger) {
		this.logger = logger;
	}

	public String getTextDatasetTrainFilename() {
		return getStringProperty("textdataset_train");
	}

	public String getTextDatasetTestFilename() {
		return getStringProperty("textdataset_test");
	}

	public String getAdditionalDatasetFilename() {
		return getStringProperty("additionaldataset");
	}

	public Integer getNoTopics(int defaultValue) {
		return noTopics < 0 ? getInteger("topics",defaultValue) : noTopics;
	}
	
	public Double getAlpha(double defaultValue) {
		return getDouble("alpha",defaultValue);
	}
	
	public Double getBeta(double defaultValue) {
		return getDouble("beta",defaultValue);
	}
	
	public Integer getNoIterations(int defaultValue) {
		return getInteger("iterations",defaultValue);
	}
	
	public Integer getNoBatches(int defaultValue) {
		return getInteger("batches",defaultValue);
	}
	
	public Integer getRareThreshold(int defaultValue) {
		return getInteger("rare_threshold",defaultValue);
	}
	
	public Integer getTopicInterval(int defaultValue) {
		return getInteger("topic_interval",defaultValue);
	}
	
	public Integer getStartDiagnostic(int defaultValue) {
		return getInteger("start_diagnostic",defaultValue);
	}

	/**
	 * @param seedDefault
	 * @return the seed set in the config file or the LSB of the current time if set to -1
	 */
	public int getSeed(int seedDefault) {
		int seed = getInteger("seed",seedDefault);
		if(seed==0) {seed=(int)System.currentTimeMillis();};
		return seed;
	}

	public boolean getDebug() {
		return (getStringProperty("debug")!=null) && 
				(getStringProperty("debug").equalsIgnoreCase("true") || getStringProperty("debug").equals("1"));
	}

	public boolean getPrintPhi() {
		return (getStringProperty("print_phi")!=null) && 
				(getStringProperty("print_phi").equalsIgnoreCase("true") || getStringProperty("print_phi").equals("1"));
	}

	
	public boolean getMeasureTiming() {
		return (getStringProperty("measure_timing")!=null) && 
				(getStringProperty("measure_timing").equalsIgnoreCase("true") || getStringProperty("measure_timing").equals("1"));
	}

	
	public void setNoTopics(int newValue) {
		noTopics = newValue;
	}

	
	public int getResultSize(int resultsSizeDefault) {
		return getInteger("results_size",resultsSizeDefault);
	}

	
	public String getDocumentBatchBuildingScheme(String batchBuildSchemeDefault) {
		String configProperty = getStringProperty("batch_building_scheme");
		return (configProperty == null) ? batchBuildSchemeDefault : configProperty;
	}

	
	public String getTopicBatchBuildingScheme(String batchBuildSchemeDefault) {
		String configProperty = getStringProperty("topic_batch_building_scheme");
		return (configProperty == null) ? batchBuildSchemeDefault : configProperty;
	}

	
	public double getDocPercentageSplitSize() {
		return getDouble("percentage_split_size_doc",1.0);
	}
	
	
	public double getTopicPercentageSplitSize() {
		return getDouble("percentage_split_size_topic",1.0);
	}

	
	public Integer getNoTopicBatches(int defaultValue) {
		return getInteger("topic_batches",defaultValue);
	}

	
	public String getTopicIndexBuildingScheme(String topicIndexBuildSchemeDefault) {
		String configProperty = getStringProperty("topic_index_building_scheme");
		return (configProperty == null) ? topicIndexBuildSchemeDefault : configProperty;

	}

	
	public int getInstabilityPeriod(int defaultValue) {
		return getInteger("instability_period",defaultValue);
	}

	
	public double[] getFixedSplitSizeDoc() {
		return ConfigurationUtils.getDoubleArrayProperty(this, "fixed_split_size_doc");
	}

	
	public int getFullPhiPeriod(int defaultValue) {
		return getInteger("full_phi_period",defaultValue);	}

	
	public String[] getSubTopicIndexBuilders(int i) {
		return getStringArrayProperty("sub_topic_index_builders");
	}

	
	public double topTokensToSample(double defaultValue) {
		return getDouble("percent_top_tokens",defaultValue);
	}

	
	public int[] getPrintNDocsInterval() {
		int [] defaultVal = {-1};
		return getIntArrayProperty("print_ndocs_interval", defaultVal);
	}

	
	public int getPrintNDocs() {
		return getInteger("print_ndocs_cnt",0);
	}

	
	public int[] getPrintNTopWordsInterval() {
		int [] defaultVal = {-1};
		return getIntArrayProperty("print_ntopwords_interval", defaultVal);
	}

	
	public int getPrintNTopWords() {
		return getInteger("print_ntopwords_cnt",0);
	}

	
	public int getProportionalTopicIndexBuilderSkipStep() {
		return getInteger("proportional_ib_skip_step",1);
	}

	
	public boolean logTypeTopicDensity(boolean logTypeTopicDensityDefault) {
		return (getStringProperty("log_type_topic_density")!=null) && 
				(getStringProperty("log_type_topic_density").equalsIgnoreCase("true") 
						|| getStringProperty("log_type_topic_density").equals("1"));
	}

	public boolean logDocumentDensity(boolean logDocumentDensityDefault) {
		return (getStringProperty("log_document_density")!=null) && 
				(getStringProperty("log_document_density").equalsIgnoreCase("true") 
						|| getStringProperty("log_document_density").equals("1"));
	}

	public String getExperimentOutputDirectory(String defaultDir) {
		String dir = getStringProperty("experiment_out_dir");
		if(dir != null && dir.endsWith("/")) dir = dir.substring(0,dir.length()-1);
		return (dir == null) ? defaultDir : dir;
	}

	public double getVariableSelectionPrior(double vsPriorDefault) {
		return getDouble("variable_selection_prior",vsPriorDefault);
	}

	public boolean logPhiDensity(String logPhiDensityDefault) {
		return (getStringProperty("log_phi_density")!=null) && 
				(getStringProperty("log_phi_density").equalsIgnoreCase("true") 
						|| getStringProperty("log_phi_density").equals("1"));
	}

	@Override
	public String getSamplerClass(String modelDefault) {
		String configProperty = getStringProperty("sampler_class");
		return (configProperty == null) ? modelDefault : configProperty;
	}

	@Override
	public LoggingUtils getLoggingUtil() {
		return logger;
	}

	@Override
	public String getDatasetFilename() {
		throw new IllegalStateException("Use the specific access methods for text/other data!");
	}

	@Override
	public String getScheme() {
		return null;
	}
	
	@Override
	public DOLDADataSet loadCombinedTestSet() throws IOException {
		String textdataset_fn = getTextDatasetTestFilename(); 
		DataSet testDataSet =  super.loadTestSet();
		System.out.println("Loading text data from file: " + textdataset_fn);
		
		String stoplistFn = getStringProperty("stoplist");
		if(!haveFilename(stoplistFn)) {
			stoplistFn = "stoplist.txt";
		}
		if(haveFilename(textdataset_fn)) {
			testInstances = LDAUtils.loadInstances(textdataset_fn, 
					stoplistFn, getRareThreshold(DOLDAConfiguration.RARE_WORD_THRESHOLD));
			if(testDataSet!=null && testInstances.size()!=testDataSet.getY().length) {
				throw new IllegalArgumentException(
						"Text data and additional covariates does not have the same length: " + testInstances.size() + " != " + testDataSet.getY().length);
			}

			List<String> ldaLabels = DOLDAConfigUtils.extractLDALabels(testInstances);
			List<String> ldaIds = DOLDAConfigUtils.extractLDAIds(testInstances);

			// If we did not have any additional covariates use the labels from the LDA dataset
			if(testDataSet==null) {
				int [] ys = new int[ldaLabels.size()];
				String [] plotLabels = null;
				Map<String,Integer> labelToId = new HashMap<>();
				Map<Integer, String> idToLabels = new HashMap<>();

				int idx = 0;
				int lblcnt = 0;
				for (int i = 0; i < ldaLabels.size(); i++) {
					String lbl = ldaLabels.get(i);
					if(labelToId.get(lbl.toString())==null) {
						labelToId.put(lbl.toString(),lblcnt);
						idToLabels.put(lblcnt++,lbl.toString());
					}
					ys[idx++] = labelToId.get(lbl.toString());
				}
				double [][] xs = new double[ldaLabels.size()][0];
				if(addIntercept) xs = MatrixOps.addIntercept(xs);
				String [] ids = new String[ldaIds.size()];
				ldaIds.toArray(ids);

				testSet = new DataSet(null, xs, ys, plotLabels, labelToId, idToLabels, null, ids);
			}
		}
		
		// Ensure that the ordering of classlabels from our loading of data
		// matches that of how MALLET sees them
		if(testInstances!=null && testSet!=null) {
			ensureMatchingClasslabels(testSet, testInstances);
		}

		
		return new DOLDADataSet(testSet, testInstances);
	}

	void ensureMatchingClasslabels(DataSet dataSet, InstanceList instanceList) {
		Map<Integer,String> lblMap = dataSet.idToLabels;
		LabelAlphabet labelAlphabet = (LabelAlphabet) instanceList.getTargetAlphabet();
		for (int c = 0; c < lblMap.size(); c++) {
			String labelName = labelAlphabet.lookupLabel(c).toString();
			//System.out.println(labelName + " =? " + lblMap.get(c));
			if(!labelName.equals(lblMap.get(c))) {
				throw new IllegalStateException("Labels from Additional Covariates does not match with labels from Text Data! " + labelName + " != " + lblMap.get(c));
			}
		}
	}

	@Override
	public DOLDADataSet loadCombinedTrainingSet() throws IOException {
		// Check if we have additional covariates and if so load them 
		String csvFilename = null;
		if(parsedCommandLine.getArgs().length>0) {
			csvFilename = parsedCommandLine.getArgs()[0];
		} else {
			csvFilename = getAdditionalDatasetTrainFilename();
			if(csvFilename !=null && csvFilename.length()==0) {
				csvFilename = null;
			}
		}
		String textdataset_fn = getTextDatasetTrainFilename();
		
		if(!haveFilename(textdataset_fn) && !haveFilename(csvFilename)) { 
			throw new IllegalArgumentException("Neither text nor additional covariates specified in configuration");
		}
				
		DataSet trainingDataSet = null;
		if(csvFilename!=null) {
			trainingDataSet = super.loadTrainingSet();
		} else {
			// Even if we don't have extra covariates we have to check if we should use intercept...
			if(configHasProperty("intercept")) {
				addIntercept = getBooleanProperty("intercept");	
				if(addIntercept) System.out.println("Using intercept...");
			}
			//Command line overrides config...
			if (parsedCommandLine.hasOption( "intercept" )) {
				int icept = Integer.parseInt(parsedCommandLine.getOptionValue("intercept").trim());
				if(icept!=0 && icept!=1 ) 
					throw new IllegalArgumentException("Intercept must be 0 (false) or 1 (true),  " + icept + " is not a legal value");
				addIntercept = icept == 1;
				if(addIntercept) System.out.println("Using intercept...");
			}
		}
		
		String stoplistFn = getStoplistFilename("stoplist.txt");
		boolean fakeTextData = false;
		if(haveFilename(textdataset_fn)) {
			trainingInstances = LDAUtils.loadInstancesPrune(textdataset_fn, 
					stoplistFn, getRareThreshold(DOLDAConfiguration.RARE_WORD_THRESHOLD),keepNumbers());
		} else {
			// Create an empty text dataset 
			trainingInstances = createEmptyTrainingset(trainingDataSet.getLabels(),trainingDataSet.getIds());
			fakeTextData = true;
		}
		
		if(trainingDataSet!=null && trainingInstances!=null && trainingInstances.size()!=trainingDataSet.getY().length) {
			throw new IllegalArgumentException(
					"Text data and additional covariates does not have the same length: " + trainingInstances.size() + " != " + trainingDataSet.getY().length);
		}
		
		// If we did not have any additional covariates use the labels from the LDA dataset
		if(trainingDataSet==null && trainingInstances!=null) {
			List<String> ldaLabels = DOLDAConfigUtils.extractLDALabels(trainingInstances);
			List<String> ldaIds = DOLDAConfigUtils.extractLDAIds(trainingInstances);

			int [] ys = new int[ldaLabels.size()];
			String [] plotLabels = null;
			Map<String,Integer> labelToId = new HashMap<>();
			Map<Integer, String> idToLabels = new HashMap<>();
			
			int idx = 0;
			int lblcnt = 0;
			for (int i = 0; i < ldaLabels.size(); i++) {
				String lbl = ldaLabels.get(i);
				if(labelToId.get(lbl.toString())==null) {
					labelToId.put(lbl.toString(),lblcnt);
					idToLabels.put(lblcnt++,lbl.toString());
				}
				ys[idx++] = labelToId.get(lbl.toString());
			}
			double [][] xs = new double[ldaLabels.size()][0];
			if(addIntercept) xs = MatrixOps.addIntercept(xs);
			String [] ids = new String[ldaIds.size()];
			ldaIds.toArray(ids);
			
			trainingSet = new DataSet(null, xs, ys, plotLabels, labelToId, idToLabels, null, ids);
		}
		
		// Ensure that the ordering of classlabels from our loading of data
		// matches that of how MALLET sees them
		if(trainingInstances!=null && trainingSet!=null) {
			ensureMatchingClasslabels(trainingSet, trainingInstances);
		}
		
		return new DOLDADataSet(trainingSet, trainingInstances, fakeTextData);
	}

	private InstanceList createEmptyTrainingset(String [] labels, String [] ids) {
		Alphabet alphabet = new Alphabet(0);
		LabelAlphabet fakeTargetAlphabet = new LabelAlphabet();

		if(ids==null) {
			ids = new String[labels.length];
			for (int i = 0; i < ids.length; i++) {
				ids[i] = "id_" + 0;
			}
		}

		for (String label : labels) {
			fakeTargetAlphabet.lookupIndex(label, true);
		}

		InstanceList fakeinstances = new InstanceList(alphabet, fakeTargetAlphabet);
		fakeinstances.addThruPipe(new EmptyInstanceIterator(labels, ids, fakeTargetAlphabet, alphabet));

		return fakeinstances;
	}

	@Override
	public InstanceList getTrainTextData() {
		return trainingInstances;
	}

	@Override
	public InstanceList getTestTextData() {
		return testInstances;
	}

	@Override
	public Integer getNoSupervisedTopics(int defaultValue) {
		return noSupervisedTopics < 0 ? getInteger("supervised_topics",defaultValue) : noSupervisedTopics;	
	}
	
	@Override
	public void setNoSupervisedTopics(int newValue) {
		noSupervisedTopics = newValue;
	}

	@Override
	public int getNoXFolds(int defaultNoFolds) {
		return getInteger("x_folds",defaultNoFolds);
	}
	
	@Override
	public int getNoTestIterations(int defaultValue) {
		return getInteger("test_iterations",defaultValue);
	}

	@Override
	public String getTopicPriorFilename() {
		return getStringProperty("topic_prior_filename");
	}

	@Override
	public String getStoplistFilename(String defaultStoplist) {
		String stoplistFn = getStringProperty("stoplist");
		if(stoplistFn==null || stoplistFn.trim().length()==0) {
			stoplistFn = defaultStoplist;
		}
		return stoplistFn;
	}

	@Override
	public boolean keepNumbers() {
		return (getStringProperty("keep_numbers")!=null) && 
				(getStringProperty("keep_numbers").equalsIgnoreCase("true") || getStringProperty("keep_numbers").equals("1"));
	}

	public boolean saveDocumentTopicMeans() {
		String key = "save_doc_topic_means";
		return (getStringProperty(key)!=null) && 
				(getStringProperty(key).equalsIgnoreCase("true") || getStringProperty(key).equals("1"));

	}

	@Override
	public String getDocumentTopicMeansOutputFilename() {
		return getStringProperty("doc_topic_mean_filename");
	}
	
	@Override
	public String getPhiMeansOutputFilename() {
		return getStringProperty("phi_mean_filename");
	}

	@Override
	public boolean savePhiMeans(boolean defaultVal) {
		String key = "save_phi_mean";
		Object prop = super.getProperty(translateKey(key));
		if(prop==null) return defaultVal;
		return getBooleanProperty(key);
	}

	@Override
	public int getPhiBurnInPercent(int phiBurnInDefault) {
		return getInteger("phi_mean_burnin", phiBurnInDefault);
	}

	@Override
	public int getPhiMeanThin(int phiMeanThinDefault) {
		return getInteger("phi_mean_thin", phiMeanThinDefault);
	}
	
	@Override
	public UncaughtExceptionHandler getUncaughtExceptionHandler() {
		return eh;
	}

	@Override
	public void setUncaughtExceptionHandler(UncaughtExceptionHandler eh) {
		this.eh = eh;
	}

	@Override
	public boolean getKeepConnectingPunctuation(boolean defaultKeepConnectingPunctuation) {
		String key = "keep_connecting_punctuation";
		Object prop = super.getProperty(translateKey(key));
		if(prop==null) return defaultKeepConnectingPunctuation;
		return getBooleanProperty(key);
	}

	@Override
	public int getMaxDocumentBufferSize(int defaltSize) {
		return getInteger("max_doc_buf_size",defaltSize);
	}

	@Override
	public int getNrTopWords(int defaltNo) {
		return getInteger("nr_top_words",defaltNo);
	}

	@Override
	public Integer getTfIdfVocabSize(int defaultValue) {
		return getInteger("tfidf_vocab_size",defaultValue);
	}
	
	@Override
	public boolean saveVocabulary(boolean defaultValue) {
		String key = "save_vocabulary";
		Object prop = super.getProperty(translateKey(key));
		if(prop==null) return defaultValue;
		return getBooleanProperty(key);
	}
	
	@Override
	public String getVocabularyFilename() {
		return getStringProperty("vocabulary_filename");
	}

	@Override
	public boolean saveTermFrequencies(boolean defaultValue) {
		String key = "save_term_frequencies";
		Object prop = super.getProperty(translateKey(key));
		if(prop==null) return defaultValue;
		return getBooleanProperty(key);
	}

	@Override
	public String getTermFrequencyFilename() {
		return getStringProperty("term_frequencies_filename");
	}

	@Override
	public boolean saveDocLengths(boolean defaultValue) {
		String key = "save_doc_lengths";
		Object prop = super.getProperty(translateKey(key));
		if(prop==null) return defaultValue;
		return getBooleanProperty(key);
	}

	@Override
	public String getDocLengthsFilename() {
		return getStringProperty("doc_lengths_filename");
	}
	
	@Override
	public double getLambda(double defaultValue) {
		return getDouble("lambda",defaultValue);
	}
	
	@Override
	public String getDocumentTopicThetaOutputFilename() {
		return getStringProperty("doc_topic_theta_filename");
	}

	@Override
	public boolean saveDocumentThetaEstimate() {
		String key = "save_doc_theta_estimate";
		return getBooleanProperty(key);
	}

	@Override
	public String getDocumentTopicDiagnosticsOutputFilename() {
		return getStringProperty("doc_topic_diagnostics_filename");
	}

	@Override
	public String getFileRegex(String string) {
		String ext = getStringProperty("file_regex");
		return (ext == null || ext.length() == 0) ? string : ext;
	}

	@Override
	public String getTestDatasetFilename() {
		return getStringProperty("test_dataset");
	}

	@Override
	public boolean saveDocumentTopicDiagnostics() {
		String key = "save_doc_topic_diagnostics";
		return getBooleanProperty(key);
	}

	@Override
	public String likelihoodType() {
		String key = "log_likelihood_type";
		return getStringProperty(key);
	}
}
