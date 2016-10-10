package xyz.lejon.configuration;

import java.io.IOException;

import xyz.lejon.utils.DataSet;
import xyz.lejon.utils.LoggingUtils;


public interface DOConfiguration {
	
	public static String PROGRAM_NAME = "DOProbit";
	public static int ITERATIONS_DEFAULT = 200;
	public static String MODEL_DEFAULT = "xyz.lejon.bayes.models.probit.SerialDOSampler";
	
	public DataSet loadTrainingSet() throws IOException;
	
	public DataSet loadTestSet() throws IOException;

	public void setLoggingUtil(LoggingUtils lu);

	public String[] getSubConfigs();

	public void activateSubconfig(String conf);

	public String whereAmI();

	public String getDatasetFilename();

	public int getLag();

	public int getBurnIn();

	public Integer getNoIterations(int iterationsDefault);

	public boolean getSaveBetas();

	public String betasOutputFn();

	public boolean doPlot();

	public String getSamplerClass(String modelDefault);

	public void forceActivateSubconfig(String string);

	public boolean saveBetaSamples();

	public String betaSamplesOutputFn();
	
	boolean getUseIntercept();
}