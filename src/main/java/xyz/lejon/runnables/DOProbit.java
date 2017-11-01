package xyz.lejon.runnables;

import java.io.File;
import java.io.PrintWriter;
import java.util.List;
import java.util.Map;

import xyz.lejon.bayes.models.dolda.DOLDAEvaluation;
import xyz.lejon.bayes.models.probit.DOEvaluation;
import xyz.lejon.bayes.models.probit.DOSampler;
import xyz.lejon.bayes.models.probit.ModelFactory;
import xyz.lejon.configuration.ConfigFactory;
import xyz.lejon.configuration.Configuration;
import xyz.lejon.configuration.DOCommandLineParser;
import xyz.lejon.configuration.DOConfiguration;
import xyz.lejon.configuration.ParsedDOConfiguration;
import xyz.lejon.eval.EvalResult;
import xyz.lejon.utils.ClassificationResultPlot;
import xyz.lejon.utils.DataSet;
import xyz.lejon.utils.LoggingUtils;
import xyz.lejon.utils.MatrixOps;

public class DOProbit {
	
	private static final String PROGRAM_NAME = DOConfiguration.PROGRAM_NAME;
	static boolean saveConfusionMatrixAsCsv = true;

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
		/*String buildVer = LoggingUtils.getManifestInfo("Implementation-Build","PCPLDA");
		String implVer  = LoggingUtils.getManifestInfo("Implementation-Version", "PCPLDA");
		if(buildVer==null||implVer==null) {
			System.out.println("GIT info:" + LoggingUtils.getLatestCommit());
		} else {
			System.out.println("Build info:" 
					+ "Implementation-Build = " + buildVer + ", " 
					+ "Implementation-Version = " + implVer);
		}*/
		
		DOCommandLineParser cp = new DOCommandLineParser(args);
		// We have to create this temporary config because at this stage if we want to create a new config for each run
		ParsedDOConfiguration tmpconfig = (ParsedDOConfiguration) ConfigFactory.getMainConfiguration(cp);
		int numberOfRuns = tmpconfig.getInt("no_runs");
		System.out.println("Doing: " + numberOfRuns + " runs");
		String expDir = tmpconfig.getExperimentOutputDirectory("");
		if(!expDir.equals("")) {
			if(!expDir.endsWith("/")) {
				expDir += "/";
			}
		}
		String logSuitePath = "Runs/" + expDir + "RunSuite" + LoggingUtils.getDateStamp();

		for (int i = 0; i < numberOfRuns; i++) {
			System.out.println("Starting run: " + i);
			DOConfiguration config = (DOConfiguration) ConfigFactory.getMainConfiguration(cp);
			LoggingUtils lu = new LoggingUtils();
			lu.checkAndCreateCurrentLogDir(logSuitePath);
			config.setLoggingUtil(lu);

			String [] configs = config.getSubConfigs();
			for(String conf : configs) {
				lu.checkCreateAndSetSubLogDir(conf);
				config.activateSubconfig(conf);
				LoggingUtils.doInitialLogging(cp, (Configuration) config, DOProbit.class.getName(), "initial_log", null, lu.getLogDir().getAbsolutePath());
				
				System.out.println("Running SubConf: " + conf);
				System.out.println("Using Config: " + config.whereAmI());
				String dataset_fn = config.getTrainingsetFilename();
				System.out.println("Using dataset: " + dataset_fn);
				DataSet trainingSetData = config.loadTrainingSet();
				//System.out.println("Design matrix covariates: " + Arrays.asList(trainingSetData.getColnamesX()));
				DataSet testSetData = config.loadTestSet();
				System.out.println("Using lag: " + config.getLag());
				System.out.println("Using burnIn: " + config.getBurnIn());

				double [][] xs = trainingSetData.getX();
				int [] ys = trainingSetData.getY();
				String [] labels = trainingSetData.getLabels();
				Map<String,Integer> labelMap = trainingSetData.getLabelToId();
				Map<Integer,String> idMap = trainingSetData.getIdToLabels();

				DOSampler doProbit = ModelFactory.get(config, xs, ys, labelMap.size());
				
				System.out.println("Using sampler: " + doProbit.getClass().getName());
				System.out.println("X is: " + MatrixOps.doubleArrayToPrintString(xs, 5));
				
				long t1 = System.currentTimeMillis();
				doProbit.sample(config.getNoIterations(DOConfiguration.ITERATIONS_DEFAULT));
				long t2 = System.currentTimeMillis();

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
				
				System.out.println("Final Betas: " + MatrixOps.doubleArrayToPrintString(betas));
				System.out.println();
				System.out.println("Confusion Matrix: \n" + DOLDAEvaluation.confusionMatrixToString(result.confusionMatrix,idMap));
				System.out.println();
				System.out.println("Total correct: " + result.noCorrect + " / " + ((double)testset.length) +  " => " + String.format("%.0f",pCorrect) + "% correct");
				System.out.println(PROGRAM_NAME + " took: " + ((double) (t2-t1) / 1000.0) + " seconds");
				
				String[] colnamesX = trainingSetData.getColnamesX();
				String [] columnLabels = new String[colnamesX.length+1];
				columnLabels[0] = "Class";
				for (int j = 1; j < columnLabels.length; j++) {
					columnLabels[j] = colnamesX[j-1];
				}
				
				File lgDir = lu.getLogDir();
				if(saveConfusionMatrixAsCsv) {
					PrintWriter pw = new PrintWriter(lgDir.getAbsolutePath() + "/last-confusion-matrix.csv");
					pw.println(DOLDAEvaluation.confusionMatrixToCSV(result.confusionMatrix,idMap, ","));
					pw.flush();
					pw.close();
				}
				
				// Save last betas
				if(config.getSaveBetas()) {
					ExperimentUtils.saveBetas(lgDir, columnLabels, xs[0].length, betas, idMap, config.betasOutputFn());
				}
				
				// Save sampled betas if that is turned on in config
				if(config.saveBetaSamples()) {
					List<double []> [] sampledBetas = doProbit.getSampledBetas();
					ExperimentUtils.saveBetaSamples(lgDir, columnLabels, xs[0].length, sampledBetas, idMap, config.betaSamplesOutputFn());
				}
				
				if(config.doPlot()) {
					ClassificationResultPlot.plot2D(labels, xs);					
				}
			}
		}
		// Ensure that we exit even if there are non-daemon threads hanging around
		System.err.println("Finished Exiting...");
		System.exit(0);
	}
}
