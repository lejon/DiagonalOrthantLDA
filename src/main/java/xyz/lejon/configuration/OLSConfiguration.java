package xyz.lejon.configuration;

import java.io.IOException;
import java.util.Map;

import xyz.lejon.utils.LoggingUtils;
import xyz.lejon.utils.RegressionDataSet;

public interface OLSConfiguration {
	public static String PROGRAM_NAME = "Linear Regression";
	public static int ITERATIONS_DEFAULT = 200;
	public static String MODEL_DEFAULT = "xyz.lejon.bayes.models.regression.LinearRegressionJBlasNormalPrior";
	
	public RegressionDataSet loadTrainingSet() throws IOException;
	
	public RegressionDataSet loadTestSet() throws IOException;
	
	public Integer getNoIterations(int defaultValue);
	
	public LoggingUtils getLoggingUtil();

	public void setLoggingUtil(LoggingUtils logger);

	public void activateSubconfig(String subConfName);

	public void forceActivateSubconfig(String subConfName);

	public String getActiveSubConfig();

	public String[] getSubConfigs();

	public String whereAmI();

	public String getDatasetFilename();
	
	public String getTrainingsetFilename();

	public int getSeed(int seedDefault);
	
	public boolean getDebug();
	
	public int [] getIntArrayProperty(String key, int [] defaultValues);
	
	void setProperty(String key, Object value);

	public double[][] getX();

	public double[] getY();

	public String[] getLabels();

	public Map<String, Integer> getLabelMap();

	public Map<Integer, String> getIdMap();

	public boolean doSave();

	public String outputFn();

	public boolean doPlot();

	public int getLag();

	public int getBurnIn();

	public RegressionDataSet getTestSet();
	
	public RegressionDataSet getTrainingSet();

	public String getSamplerClass(String modelDefault);

	public boolean getUseIntercept();
}
