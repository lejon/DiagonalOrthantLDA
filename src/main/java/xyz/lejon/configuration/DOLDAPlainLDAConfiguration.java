package xyz.lejon.configuration;

import java.util.Map;

import org.apache.commons.configuration.ConfigurationException;

import cc.mallet.configuration.LDAConfiguration;
import cc.mallet.util.LoggingUtils;

/**
 * This class wraps a DOLDA configuration so that it looks like an LDAConfiguration
 * so we can use it with the Partially Collapsed sampler
 * 
 * @author Leif Jonsson
 *
 */
public class DOLDAPlainLDAConfiguration extends SubConfig implements LDAConfiguration {

	private static final long serialVersionUID = 1L;
	DOLDAConfiguration parent;
	LoggingUtils ldaLogger;
	

	public DOLDAPlainLDAConfiguration(DOLDAConfiguration parentCfg) throws ConfigurationException {
		super(parentCfg.whereAmI());
		this.parent = parentCfg;
		xyz.lejon.utils.LoggingUtils loggingUtil = parentCfg.getLoggingUtil();
		ldaLogger = new LoggingUtils(loggingUtil.getBaseDir(), loggingUtil.getLogDir(), loggingUtil.getTimings());
	}
	
	public String getSamplerClass(String modelDefault) {
		return parent.getSamplerClass(modelDefault);
	}

	public String getTextDatasetFilename() {
		return parent.getTextDatasetTrainFilename();
	}

	public String[] getLabels() {
		return parent.getLabels();
	}

	public Map<String, Integer> getLabelMap() {
		return parent.getLabelMap();
	}

	public Map<Integer, String> getIdMap() {
		return parent.getIdMap();
	}

	public int getNoIterations(Integer noIterDefault) {
		return parent.getNoIterations(noIterDefault);
	}

	public LoggingUtils getLoggingUtil() {
		return ldaLogger;
	}

	public void setLoggingUtil(xyz.lejon.utils.LoggingUtils logger) {
		parent.setLoggingUtil(logger);
	}
	
	@Override
	public void setLoggingUtil(LoggingUtils logger) {
		this.ldaLogger = logger;
	}		

	public void activateSubconfig(String subConfName) {
		parent.activateSubconfig(subConfName);
	}

	public void forceActivateSubconfig(String subConfName) {
		parent.forceActivateSubconfig(subConfName);
	}

	public String getActiveSubConfig() {
		return parent.getActiveSubConfig();
	}

	public String[] getSubConfigs() {
		return parent.getSubConfigs();
	}

	public String whereAmI() {
		return parent.whereAmI();
	}

	public String getDatasetFilename() {
		return parent.getTextDatasetTrainFilename();
	}

	public String getScheme() {
		return parent.getScheme();
	}

	public Integer getNoTopics(int defaultValue) {
		return parent.getNoTopics(defaultValue);
	}

	public void setNoTopics(int newValue) {
		parent.setNoTopics(newValue);
	}

	public Double getAlpha(double defaultValue) {
		return parent.getAlpha(defaultValue);
	}

	public Double getBeta(double defaultValue) {
		return parent.getBeta(defaultValue);
	}

	public Integer getNoIterations(int defaultValue) {
		return parent.getNoIterations(defaultValue);
	}

	public Integer getNoBatches(int defaultValue) {
		return parent.getNoBatches(defaultValue);
	}

	public Integer getNoTopicBatches(int defaultValue) {
		return parent.getNoTopicBatches(defaultValue);
	}

	public Integer getRareThreshold(int defaultValue) {
		return parent.getRareThreshold(defaultValue);
	}

	public Integer getTopicInterval(int defaultValue) {
		return parent.getTopicInterval(defaultValue);
	}

	public Integer getStartDiagnostic(int defaultValue) {
		return parent.getStartDiagnostic(defaultValue);
	}

	public int getSeed(int seedDefault) {
		return parent.getSeed(seedDefault);
	}

	public boolean getDebug() {
		return parent.getDebug();
	}

	public boolean getPrintPhi() {
		return parent.getPrintPhi();
	}

	public int[] getIntArrayProperty(String key, int[] defaultValues) {
		return parent.getIntArrayProperty(key, defaultValues);
	}

	public boolean getMeasureTiming() {
		return parent.getMeasureTiming();
	}

	public int getResultSize(int resultsSizeDefault) {
		return parent.getResultSize(resultsSizeDefault);
	}

	public String getDocumentBatchBuildingScheme(String batchBuildSchemeDefault) {
		return parent.getDocumentBatchBuildingScheme(batchBuildSchemeDefault);
	}

	public String getTopicBatchBuildingScheme(String batchBuildSchemeDefault) {
		return parent.getTopicBatchBuildingScheme(batchBuildSchemeDefault);
	}

	public String getTopicIndexBuildingScheme(String topicIndexBuildSchemeDefault) {
		return parent.getTopicIndexBuildingScheme(topicIndexBuildSchemeDefault);
	}

	public double getDocPercentageSplitSize() {
		return parent.getDocPercentageSplitSize();
	}

	public double getTopicPercentageSplitSize() {
		return parent.getTopicPercentageSplitSize();
	}

	public int getInstabilityPeriod(int defaultValue) {
		return parent.getInstabilityPeriod(defaultValue);
	}

	public double[] getFixedSplitSizeDoc() {
		return parent.getFixedSplitSizeDoc();
	}

	public int getFullPhiPeriod(int defaultValue) {
		return parent.getFullPhiPeriod(defaultValue);
	}

	public String[] getSubTopicIndexBuilders(int i) {
		return parent.getSubTopicIndexBuilders(i);
	}

	public double topTokensToSample(double defaultValue) {
		return parent.topTokensToSample(defaultValue);
	}

	public void setProperty(String key, Object value) {
		parent.setProperty(key, value);
	}

	public int[] getPrintNDocsInterval() {
		return parent.getPrintNDocsInterval();
	}

	public int getPrintNDocs() {
		return parent.getPrintNDocs();
	}

	public int[] getPrintNTopWordsInterval() {
		return parent.getPrintNTopWordsInterval();
	}

	public int getPrintNTopWords() {
		return parent.getPrintNTopWords();
	}

	public int getProportionalTopicIndexBuilderSkipStep() {
		return parent.getProportionalTopicIndexBuilderSkipStep();
	}

	public boolean logTypeTopicDensity(boolean logTypeTopicDensityDefault) {
		return parent.logTypeTopicDensity(logTypeTopicDensityDefault);
	}

	public boolean logDocumentDensity(boolean logDocumentDensityDefault) {
		return parent.logDocumentDensity(logDocumentDensityDefault);
	}

	public String getExperimentOutputDirectory(String defaultDir) {
		return parent.getExperimentOutputDirectory(defaultDir);
	}

	public double getVariableSelectionPrior(double vsPriorDefault) {
		return parent.getVariableSelectionPrior(vsPriorDefault);
	}

	public boolean logPhiDensity(String logPhiDensityDefault) {
		return parent.logPhiDensity(logPhiDensityDefault);
	}

	@Override
	public String getTopicPriorFilename() {
		return parent.getTopicPriorFilename();
	}

	@Override
	public String getStoplistFilename(String string) {
		return parent.getStoplistFilename(string);
	}

	@Override
	public boolean keepNumbers() {
		return parent.keepNumbers();
	}

	@Override
	public boolean saveDocumentTopicMeans() {
		return parent.saveDocumentTopicMeans();
	}

	@Override
	public String getDocumentTopicMeansOutputFilename() {
		return parent.getDocumentTopicMeansOutputFilename();
	}

	@Override
	public String getPhiMeansOutputFilename() {
		return parent.getPhiMeansOutputFilename();
	}

	@Override
	public boolean savePhiMeans(boolean defaultValue) {
		return parent.savePhiMeans(defaultValue);
	}

	@Override
	public int getPhiBurnInPercent(int phiBurnInDefault) {
		return parent.getPhiBurnInPercent(phiBurnInDefault);
	}

	@Override
	public int getPhiMeanThin(int phiMeanThinDefault) {
		return parent.getPhiMeanThin(phiMeanThinDefault);
	}

	@Override
	public boolean getKeepConnectingPunctuation(boolean arg0) {
		return parent.getKeepConnectingPunctuation(arg0);
	}

	@Override
	public int getMaxDocumentBufferSize(int arg0) {
		return parent.getMaxDocumentBufferSize(arg0);
	}

	@Override
	public int getNrTopWords(int arg0) {
		return parent.getNrTopWords(arg0);
	}

	@Override
	public Integer getTfIdfVocabSize(int arg0) {
		return parent.getTfIdfVocabSize(arg0);
	}

	@Override
	public String getDocLengthsFilename() {
		return parent.getDocLengthsFilename();
	}

	@Override
	public String getTermFrequencyFilename() {
		return parent.getTermFrequencyFilename();
	}

	@Override
	public String getVocabularyFilename() {
		return parent.getVocabularyFilename();
	}

	@Override
	public boolean saveDocLengths(boolean arg0) {
		return parent.saveDocLengths(arg0);
	}

	@Override
	public boolean saveTermFrequencies(boolean arg0) {
		return parent.saveTermFrequencies(arg0);
	}

	@Override
	public boolean saveVocabulary(boolean arg0) {
		return parent.saveVocabulary(arg0);
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
	public String getDirichletSamplerBuilderClass(String samplerBuilderClassName) {
		return LDAConfiguration.SPARSE_DIRICHLET_SAMPLER_BULDER_DEFAULT;
	}

	@Override
	public int getAliasPoissonThreshold(int aliasPoissonDefaultThreshold) {
		return LDAConfiguration.ALIAS_POISSON_DEFAULT_THRESHOLD;
	}
}
