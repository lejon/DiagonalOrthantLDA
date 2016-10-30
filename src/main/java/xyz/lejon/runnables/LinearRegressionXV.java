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
import xyz.lejon.utils.PlainCrossValidationIterator;

public class LinearRegressionXV {

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
		int runs = tmpconfig.getInt("no_runs");
		System.out.println("Doing: " + runs + " runs");
		String logSuitePath = "Runs/RunSuite" + LoggingUtils.getDateStamp();

		OLSConfiguration config = (OLSConfiguration) OLSConfigFactory.getMainConfiguration(cp);
		LoggingUtils lu = new LoggingUtils();
		lu.checkAndCreateCurrentLogDir(logSuitePath);
		config.setLoggingUtil(lu);
		
		System.out.println("Using Config: " + config.whereAmI());

		String [] configs = config.getSubConfigs();
		for(String conf : configs) {
			lu.checkCreateAndSetSubLogDir(conf);
			config.activateSubconfig(conf);
			LoggingUtils.doInitialLogging(cp, (Configuration) config, LinearRegressionXV.class.getName(), "initial_log", null, lu.getLogDir().getAbsolutePath());

			System.out.println("Running SubConf: " + conf);
			System.out.println("Using Config: " + config.whereAmI());
			String dataset_fn = config.getDatasetFilename();
			System.out.println("Using dataset: " + dataset_fn);
			config.loadTrainingSet();
			config.loadTestSet();
			System.out.println("Using lag: " + config.getLag());
			System.out.println("Using burnIn: " + config.getBurnIn());

			double [][] allXs = config.getX();
			double [] allYs = config.getY();
			
			System.out.println("X'es are: " + MatrixOps.doubleArrayToPrintString(allXs,10,10));
			System.out.println("Y's are: " + MatrixOps.arrToStr(allYs));
						
			int folds = 5;
			PlainCrossValidationIterator cvIter = new PlainCrossValidationIterator(allXs,allYs,folds);

			double [] mses = new double[folds];
			double [] genErr = new double[folds];
			long t1 = System.currentTimeMillis();
			for (int fold = 0; fold < folds; fold++) {					
				double [][] xs = cvIter.nextTrainX();
				double [] ys = cvIter.nextTrainY();
					
				System.out.println("Starting fold: " + fold);

				LinearRegression linearRegression = ModelFactory.get(config, xs, ys);

				System.out.println("Using sampler: " + linearRegression.getClass().getName());

				//linearRegression.setNoIter(true);
				linearRegression.sample(config.getNoIterations(DOConfiguration.ITERATIONS_DEFAULT));

				double [] betas = linearRegression.getBetas();
				
				double mse = evaluate(ys, xs, betas, config.getUseIntercept());
				mses[fold] = mse;
				double generalizationError = evaluate(cvIter.nextTestY(), cvIter.nextTestX(), betas, config.getUseIntercept());
				genErr[fold] = generalizationError;

				System.out.println("MSE on trainingset: " + linearRegression.getClass().getSimpleName() + " => " + mse);
				System.out.println("MSE on testset    : " + linearRegression.getClass().getSimpleName() + " => " + generalizationError);
				MatrixOps.noDigits = 10;
				String [] coeffs = config.getTrainingSet().getTransformedColNames();
				printBetaCoefficients(coeffs, betas);				

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
			long t2 = System.currentTimeMillis();
			System.out.println(PROGRAM_NAME + " took: " + ((double) (t2-t1) / 1000.0) + " seconds");
			double meanMse = 0.0;
			for (int i = 0; i < mses.length; i++) {
				meanMse += mses[i];
			}
			meanMse /= folds;
			double meanGenErr = 0.0;
			for (int i = 0; i < mses.length; i++) {
				meanGenErr += genErr[i];
			}
			meanGenErr /= folds;
			System.out.println("Mean MSE on training set evaluated on " + folds + " folds is: " + meanMse);
			System.out.println("Mean MSE on test     set evaluated on " + folds + " folds is: " + meanGenErr);
		}
		// Ensure that we exit even if there are non-daemon threads hanging around
		System.err.println("Finished! Exiting...");
		System.exit(0);
	}

	private static void printBetaCoefficients(String [] coeffs, double [] betas) {
		int longestCoeffName = 0;

		for (int j = 0; j < coeffs.length; j++) {
			if(longestCoeffName<coeffs[j].length()) longestCoeffName = coeffs[j].length();
		}
		System.out.println("Final Betas: ");
		System.out.print("Variable");
		int addSpace = 5;
		for (int k = 0; k < (longestCoeffName-"Variable".length()+addSpace); k++) {
			System.out.print(" ");
		}
		System.out.println("Estimate");

		for (int k = 0; k < (longestCoeffName+addSpace+"Estimate".length()); k++) {
			System.out.print("=");
		}
		System.out.println();
		
		for (int j = 0; j < betas.length; j++) {
			System.out.print(coeffs[j]);
			int spaceLen = longestCoeffName - coeffs[j].length();
			spaceLen += addSpace;
			for (int k = 0; k < spaceLen; k++) {
				System.out.print(" ");
			}
			System.out.println(betas[j]);
		}
		System.out.println();
	}

	static double evaluate(double[] nextTestY, double[][] nextTestX, double[] betas, boolean useIntercept) {
		double mse = 0.0;
		for (int i = 0; i < nextTestY.length; i++) {
			double y_hat = 0.0;
			for (int j = 0; j < nextTestX[i].length; j++) {
				if(j == 0 && useIntercept) { 
					y_hat += betas[j];
					continue;
				}
				y_hat += nextTestX[i][j] * betas[j];
			}
			double error = (nextTestY[i]-y_hat); 
			mse += error*error; 
		}
		return mse / nextTestY.length;
	}

}
