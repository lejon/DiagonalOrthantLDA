package xyz.lejon.configuration;

import java.io.IOException;
import java.lang.Thread.UncaughtExceptionHandler;

import org.apache.commons.configuration.ConfigurationException;

import cc.mallet.types.InstanceList;
import xyz.lejon.bayes.models.dolda.DOLDADataSet;
import xyz.lejon.utils.DataSet;
import xyz.lejon.utils.LoggingUtils;

public interface DOLDAConfiguration extends DOConfiguration {

	String MODEL_DEFAULT = "xyz.lejon.bayes.models.dolda.DOLDAGibbsEJMLHorseshoe";
	int SEED_DEFAULT = 0;
	int RARE_WORD_THRESHOLD = 0;
	Integer NO_ITER_DEFAULT = 200;
	int XFOLDS = 10;
	
	String getSamplerClass(String modelDefault);

	String getTextDatasetTrainFilename();
	
	String getTextDatasetTestFilename();
	
	public LoggingUtils getLoggingUtil();

	public void setLoggingUtil(LoggingUtils logger);

	public void activateSubconfig(String subConfName);

	public void forceActivateSubconfig(String subConfName);

	public String getActiveSubConfig();

	public String[] getSubConfigs();

	public String whereAmI();

	public String getDatasetFilename();

	public String getScheme();

	public Integer getNoTopics(int defaultValue);
	
	public Integer getNoSupervisedTopics(int defaultValue);
	
	public void setNoTopics(int newValue);

	public Double getAlpha(double defaultValue);

	public Double getBeta(double defaultValue);

	public Integer getNoIterations(int defaultValue);

	public Integer getNoBatches(int defaultValue);

	public Integer getNoTopicBatches(int defaultValue);

	public Integer getRareThreshold(int defaultValue);

	public Integer getTopicInterval(int defaultValue);

	public Integer getStartDiagnostic(int defaultValue);

	public int getSeed(int seedDefault);
	
	public boolean getDebug();

	public boolean getPrintPhi();
	
	public int [] getIntArrayProperty(String key, int [] defaultValues);
	
	public boolean getMeasureTiming();

	public int getResultSize(int resultsSizeDefault);

	public String getDocumentBatchBuildingScheme(String batchBuildSchemeDefault);

	// How to build the topic batches
	public String getTopicBatchBuildingScheme(String batchBuildSchemeDefault);
	
	// How to build which words in the topics to sample
	public String getTopicIndexBuildingScheme(String topicIndexBuildSchemeDefault);

	public double getDocPercentageSplitSize();
	
	public double getTopicPercentageSplitSize();

	public int getInstabilityPeriod(int defaultValue);

	public double[] getFixedSplitSizeDoc();

	public int getFullPhiPeriod(int defaultValue);

	public String[] getSubTopicIndexBuilders(int i);

	public double topTokensToSample(double defaultValue);
	
	void setProperty(String key, Object value);

	public int[] getPrintNDocsInterval();

	public int getPrintNDocs();

	public int[] getPrintNTopWordsInterval();

	public int getPrintNTopWords();

	public int getProportionalTopicIndexBuilderSkipStep();

	public boolean logTypeTopicDensity(boolean logTypeTopicDensityDefault);

	public boolean logDocumentDensity(boolean logDocumentDensityDefault);
	
	public String getExperimentOutputDirectory(String defaultDir);

	public double getVariableSelectionPrior(double vsPriorDefault);

	public boolean logPhiDensity(String logPhiDensityDefault);

	DataSet loadTrainingSet() throws IOException;

	DataSet loadTestSet() throws IOException;

	InstanceList getTrainTextData();
	
	InstanceList getTestTextData();

	DOLDADataSet loadCombinedTrainingSet() throws IOException, ConfigurationException;

	DOLDADataSet loadCombinedTestSet() throws IOException;

	void setNoSupervisedTopics(int newValue);

	boolean getSaveBetas();

	String betasOutputFn();

	int getNoXFolds(int defaultNoFolds);

	int getNoTestIterations(int defaultValue);

	boolean saveBetaSamples();
	
	String betaSamplesOutputFn();

	String getTopicPriorFilename();

	String getStoplistFilename(String string);
	
	boolean keepNumbers();

	boolean saveDocumentTopicMeans();

	String getDocumentTopicMeansOutputFilename();
	
	public String getPhiMeansOutputFilename();

	public boolean savePhiMeans(boolean defaultValue);

	public int getPhiBurnInPercent(int phiBurnInDefault);

	public int getPhiMeanThin(int phiMeanThinDefault);
	
	public int getBurnIn();

	int getLag();

	boolean getNormalize();

	UncaughtExceptionHandler getUncaughtExceptionHandler();

	void setUncaughtExceptionHandler(UncaughtExceptionHandler eh);
	
	public boolean getKeepConnectingPunctuation(boolean arg0);

	public int getMaxDocumentBufferSize(int arg0);

	public int getNrTopWords(int arg0);

	public Integer getTfIdfVocabSize(int arg0);

	String getVocabularyFilename();

	boolean saveVocabulary(boolean b);

	boolean saveTermFrequencies(boolean b);

	String getTermFrequencyFilename();

	boolean saveDocLengths(boolean b);

	String getDocLengthsFilename();

	double getLambda(double lambdaDefault);

	String getDocumentTopicThetaOutputFilename();

	boolean saveDocumentThetaEstimate();

	String getDocumentTopicDiagnosticsOutputFilename();

	String getFileRegex(String defaultName);

	String getTestDatasetFilename();

	boolean saveDocumentTopicDiagnostics();

	String likelihoodType();

	Integer getHyperparamOptimInterval(int defaultVal);

	boolean useSymmetricAlpha(boolean defaultVal);
}
