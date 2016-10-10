package xyz.lejon.runnables;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import joinery.DataFrame;
import xyz.lejon.bayes.models.regression.LinearRegression;
import xyz.lejon.bayes.models.regression.ModelFactory;
import xyz.lejon.configuration.Configuration;
import xyz.lejon.configuration.DOConfiguration;
import xyz.lejon.configuration.OLSCommandLineParser;
import xyz.lejon.configuration.OLSConfigFactory;
import xyz.lejon.configuration.OLSConfiguration;
import xyz.lejon.configuration.ParsedOLSConfiguration;
import xyz.lejon.utils.LoggingUtils;
import xyz.lejon.utils.MatrixOps;

public class HSLinearRegression {
	
	private static final String PROGRAM_NAME = OLSConfiguration.PROGRAM_NAME;

	public static void main(String[] args) throws Exception {

		Thread.setDefaultUncaughtExceptionHandler(new Thread.
				UncaughtExceptionHandler() {
			public void uncaughtException(Thread t, Throwable e) {
				System.out.println(t + " throws exception: " + e);
				e.printStackTrace();
				System.err.println("Main thread Exiting.");
				System.exit(-1);
			}
		});

		System.out.println("We have: " + Runtime.getRuntime().availableProcessors() 
				+ " processors avaiable");
		String buildVer = LoggingUtils.getManifestInfo("Implementation-Build","DOLDA");
		String implVer  = LoggingUtils.getManifestInfo("Implementation-Version", "DOLDA");
		if(buildVer==null||implVer==null) {
			System.out.println("GIT info:" + LoggingUtils.getLatestCommit());
		} else {
			System.out.println("Build info:" 
					+ "Implementation-Build = " + buildVer + ", " 
					+ "Implementation-Version = " + implVer);
		}
		
		OLSCommandLineParser cp = new OLSCommandLineParser(args);
		// We have to create this temporary config because at this stage if we want to create a new config for each run
		ParsedOLSConfiguration tmpconfig = (ParsedOLSConfiguration) OLSConfigFactory.getMainConfiguration(cp);
		int numberOfRuns = tmpconfig.getInt("no_runs");
		System.out.println("Doing: " + numberOfRuns + " runs");
		String logSuitePath = "Runs/RunSuite" + LoggingUtils.getDateStamp();

		for (int i = 0; i < numberOfRuns; i++) {
			System.out.println("Starting run: " + i);
			OLSConfiguration config = (OLSConfiguration) OLSConfigFactory.getMainConfiguration(cp);
			LoggingUtils lu = new LoggingUtils();
			lu.checkAndCreateCurrentLogDir(logSuitePath);
			config.setLoggingUtil(lu);

			String [] configs = config.getSubConfigs();
			for(String conf : configs) {
				lu.checkCreateAndSetSubLogDir(conf);
				config.activateSubconfig(conf);
				LoggingUtils.doInitialLogging(cp, (Configuration) config, HSLinearRegression.class.getName(), "initial_log", null, lu.getLogDir().getAbsolutePath());
				
				System.out.println("Running SubConf: " + conf);
				System.out.println("Using Config: " + config.whereAmI());
				String dataset_fn = config.getDatasetFilename();
				System.out.println("Using dataset: " + dataset_fn);
				config.loadTrainingSet();
				config.loadTestSet();
				System.out.println("Using lag: " + config.getLag());
				System.out.println("Using burnIn: " + config.getBurnIn());

				double [][] xs = config.getX();
				double [] ys = config.getY();		

				LinearRegression linearRegression = ModelFactory.get(config, xs, ys);
				
				System.out.println("Using sampler: " + linearRegression.getClass().getName());
				System.out.println("X is: " + MatrixOps.doubleArrayToPrintString(xs, 5));
				System.out.println("y is: " + MatrixOps.arrToStr(ys, 10));
				
				long t1 = System.currentTimeMillis();
				linearRegression.sample(config.getNoIterations(DOConfiguration.ITERATIONS_DEFAULT));
				long t2 = System.currentTimeMillis();

				double [] betas = linearRegression.getBetas();
				
				/*double [][] testset;
				RegressionDataSet testSetData = config.getTestSet();
				if(testSetData==null) {
					testset = xs;
				} else {
					testset = testSetData.getX();
				}*/

				//EvalResult result  = linearRegression.evaluate(testset, betas);
				//double pCorrect = ((double)result.noCorrect)/testset.length * 100;
				
				System.out.println("Result for: " + linearRegression.getClass().getName());
				MatrixOps.noDigits = 10;
				System.out.println("Final Betas: " + MatrixOps.arrToStr(betas));
				System.out.println();
				System.out.println(PROGRAM_NAME + " took: " + ((double) (t2-t1) / 1000.0) + " seconds");
				
				if(config.doSave()) {
					File lgDir = lu.getLogDir();
					String [] columnLabels = new String[betas.length+1];
					columnLabels[0] = "Class";
					for (int lblIdx = 1; lblIdx < columnLabels.length; lblIdx++) {
						// In the output name colums Xn for X covariates and Zn for supervised topics
						columnLabels[lblIdx] = "X" + lblIdx;
					}
					DataFrame<Object> out = new DataFrame<>(columnLabels);
					List<Object> row = new ArrayList<>();
					//Add the class id to the first column
					row.add("lm");
					for (int k = 0; k < betas.length; k++) {
						row.add(betas[k]);
					}
					out.append(row);
					out.writeCsv(lgDir.getAbsolutePath() + "/" +config.outputFn());

					out = new DataFrame<>(columnLabels);
					List<double []>  sampledBetas = linearRegression.getSampledBetas();
					System.out.println("Sampled betas #" + sampledBetas.size());
					for (int j = 0; j < sampledBetas.size(); j++) {
						double [] betaRow = sampledBetas.get(j);
						row = new ArrayList<>();
						//Add the class id to the first column
						row.add("lm");
						for (int covariate = 0; covariate < betaRow.length; covariate++) {
							row.add(betaRow[covariate]);
						}
						out.append(row);
					}
					out.writeCsv(lgDir.getAbsolutePath() + "/" +"sampled-" + config.outputFn());

				}
				
				PrintWriter pw = new PrintWriter("linear-regression.csv");
				for (int j = 0; j < betas.length; j++) {
					pw.print(betas[j] + ",");
				}
				pw.flush();
				pw.close();
			}
		}
		// Ensure that we exit even if there are non-daemon threads hanging around
		System.err.println("Finished Exiting...");
		System.exit(0);
	}

}
