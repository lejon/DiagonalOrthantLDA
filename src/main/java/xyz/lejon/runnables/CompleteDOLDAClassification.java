package xyz.lejon.runnables;

// To run with Pure Java implementation of BLAS, LAPACK and ARPACK use the below command line
// If this is done, you can use the DOLDAGibbsJBlasHorseshoePar to parallelize the Beta sampling (per class)
// On the mozilla-25000 dataset, this reduced the sampling of 10 iterations from 7 to 4 minutes!
// java -Xmx8g -Dcom.github.fommil.netlib.BLAS=com.github.fommil.netlib.F2jBLAS -Dcom.github.fommil.netlib.LAPACK=com.github.fommil.netlib.F2jLAPACK -Dcom.github.fommil.netlib.ARPACK=com.github.fommil.netlib.F2jARPACK -cp DOLDA-0.6.2.jar xyz.lejon.runnables.DOLDAClassification --normalize --run_cfg=DOLDAConfigs/DOLDAClassification.cfg

import java.io.File;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import xyz.lejon.bayes.models.dolda.DOLDA;
import xyz.lejon.bayes.models.dolda.DOLDAClassifier;
import xyz.lejon.bayes.models.dolda.DOLDADataSet;
import xyz.lejon.bayes.models.dolda.DOLDAPointClassifier;
import xyz.lejon.configuration.ConfigFactory;
import xyz.lejon.configuration.Configuration;
import xyz.lejon.configuration.DOLDACommandLineParser;
import xyz.lejon.configuration.DOLDAConfiguration;
import xyz.lejon.configuration.ParsedDOLDAConfiguration;
import xyz.lejon.utils.EclipseDetector;
import xyz.lejon.utils.EnhancedConfusionMatrix;
import xyz.lejon.utils.LoggingUtils;
import xyz.lejon.utils.MatrixOps;
import xyz.lejon.utils.Timer;
import cc.mallet.classify.Trial;
import cc.mallet.util.LDAUtils;
import cc.mallet.types.InstanceList;

/** 
 * A hack class for running a completing fold if the last fold crashed
 * 
 * @author Leif Jonsson
 *
 */
public class CompleteDOLDAClassification {

	public static Thread.UncaughtExceptionHandler exHandler;
	public static String PROGRAM_NAME = "CompleteDOLDAClassification";
	public static PrintWriter pw;
	protected static volatile boolean abort = false;
	protected static volatile boolean normalShutdown = false;

	static DOLDAClassifier dcls;
	static boolean saveConfusionMatrixAsCsv = true;

	private static DOLDAClassifier getCurrentSampler() {
		return dcls;
	}

	public static void main(String[] args) throws Exception {
		CompleteDOLDAClassification dc = new CompleteDOLDAClassification();
		dc.execute(args);
	}

	public void execute(String[] initargs) throws Exception {

		List<String> trainingIds = new ArrayList<String>();
		int trainFileCnt = 0;
		// Find the files which contains the ids of training observations
		for (int i = 0; i < initargs.length; i++) {
			if(initargs[i].contains("test-ids-fold-")) {
				trainFileCnt++;
				List<String> lines = Files.readAllLines(Paths.get(initargs[i]), Charset.defaultCharset());
				trainingIds.addAll(lines);
			}
		}
		
		if(trainingIds.size()==0) throw new IllegalArgumentException("Got no training ids!");

		String [] args = new String[initargs.length-trainFileCnt];
		for (int i = 0; i < args.length; i++) {
			args[i] = initargs[i];
		}

		if(args.length == 0) {
			System.out.println("\n" + PROGRAM_NAME + ": No args given, you should typically call it along the lines of: \n" 
					+ "java -cp DOLDA-X.X.X.jar xyz.lejon.runnables.CompleteDOLDAClassification --run_cfg=src/main/resources/configuration/DOLDAConfig.cfg\n" 
					+ "or\n" 
					+ "java -jar DOLDA-X.X.X.jar —run_cfg=src/main/resources/configuration/DOLDAConfig.cfg\n");
			System.exit(-1);
		}

		String [] newArgs = EclipseDetector.runningInEclipse(args);
		// If we are not running in eclipse we can install the abort functionality
		if(newArgs==null) {
			final Thread mainThread = Thread.currentThread();
			Runtime.getRuntime().addShutdownHook(new Thread() {
				public void run() {
					int waitTimeout = 4000;
					if(!normalShutdown) {
						System.err.println("Running shutdown hook: DOLDAClassifier Aborted! Waiting for shudown...");
						abort = true;
						if(getCurrentSampler()!=null) {
							getCurrentSampler().abort();
							try {
								mainThread.join(waitTimeout);
							} catch (InterruptedException e) {
								System.err.println("Exception during Join..");
								e.printStackTrace();
							}
						}
					} 
				}
			});
			// Else don't install it, but set args to be the one with "-runningInEclipse" removed
		} else {
			args = newArgs;
		}

		exHandler = new Thread.UncaughtExceptionHandler() {
			public void uncaughtException(Thread t, Throwable e) {
				System.out.println(t + " throws exception: " + e);
				e.printStackTrace();
				if(pw != null) {
					try {
						e.printStackTrace(pw);
						pw.close();
					} catch (Exception e1) {
						// Give up!
					}
				}
				System.err.println("Main thread Exiting.");
				System.exit(-1);
			}
		};

		Thread.setDefaultUncaughtExceptionHandler(exHandler);

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

		DOLDACommandLineParser cp = new DOLDACommandLineParser(args);

		// We have to create this temporary config because at this stage if we want to create a new config for each run
		ParsedDOLDAConfiguration tmpconfig = (ParsedDOLDAConfiguration) ConfigFactory.getMainConfiguration(cp);			

		String expDir = tmpconfig.getExperimentOutputDirectory("");
		if(!expDir.equals("")) {
			if(!expDir.endsWith("/")) {
				expDir += "/";
			}
		}
		String logSuitePath = "Runs/" + expDir + "RunSuite" + LoggingUtils.getDateStamp();
		// Reading in command line parameters		
		System.out.println("Starting run: " + 1);

		DOLDAConfiguration config = (DOLDAConfiguration) ConfigFactory.getMainConfiguration(cp);
		config.setUncaughtExceptionHandler(exHandler);
		LoggingUtils lu = new LoggingUtils();
		System.out.println("Logging to: " + logSuitePath);
		lu.checkAndCreateCurrentLogDir(logSuitePath);
		config.setLoggingUtil(lu);

		String [] configs = config.getSubConfigs();
		for(String conf : configs) {
			if(abort) break;
			lu.checkCreateAndSetSubLogDir(conf);
			config.activateSubconfig(conf);

			LoggingUtils.doInitialLogging(cp, (Configuration) config, this.getClass().getName(), "initial_log", null, lu.getLogDir().getAbsolutePath());

			File lgDir = lu.getLogDir();

			pw = new PrintWriter(new File(lgDir.getAbsolutePath() + "/" + PROGRAM_NAME + "-crash.txt"));

			System.out.println("Using Config: " + config.whereAmI());
			System.out.println("Runnin subconfig: " + conf);
			String dataset_fn = config.getTextDatasetTrainFilename();
			System.out.println("Using dataset: " + dataset_fn);

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

			dcls = getSamplingClassifier(config, trainingSetData);
			dcls.setFixedTrainingIds(trainingIds);

			// Just fix last fold
			int folds = 1;

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
			System.out.println(PROGRAM_NAME + " cross validation took: " + (t.getEllapsedTime() / 1000) + " seconds");

			if(saveConfusionMatrixAsCsv) {
				PrintWriter pw = new PrintWriter(lgDir.getAbsolutePath() + "/last-confusion-matrix.csv");
				pw.println(combinedConfusionMatrix.toCsv(","));
				pw.flush();
				pw.close();
			}

			/*
				if(config.doPlot()) {
					ClassificationResultPlot.plot2D(labels, xs);					
				}
			 */

			// Save file with summary of results and metadata 
			List<String> metadata = new ArrayList<String>();
			if(dolda.getAbort()) {
				metadata.add("!!!! WARNING !!!! sampling was aborted. Iterations sampled: " + dolda.getCurrentIteration());
			}
			metadata.add("No. Topics: " + dolda.getNoTopics());
			metadata.add("No. Supervised Topics: " + config.getNoSupervisedTopics(-1));
			metadata.add("Start Seed: " + dolda.getStartSeed());
			metadata.add("Accuracy: " + String.format("%.0f",(xvalidationAverage*100)));
			metadata.add("ConfusionMatrix: " + "\n" + combinedConfusionMatrix);
			// Save stats for this run
			lu.dynamicLogRun("Runs", t, cp, (Configuration) config, null, 
					this.getClass().getName(), this.getClass().getSimpleName() + "-results", "HEADING", "DOLDA", 1, metadata);

			PrintWriter out = new PrintWriter(lgDir.getAbsolutePath() + "/TopWords.txt");
			if(textData!=null) {
				String topWords = LDAUtils.formatTopWords(dolda.getTopWords(50));
				out.println(topWords);
				System.out.println("Top words are: \n" + topWords);
			} else { 
				out.println("No text data used");
			}
			out.flush();
			out.close();

			System.out.println("I am done!");
		}
		if(buildVer==null||implVer==null) {
			System.out.println("GIT info:" + LoggingUtils.getLatestCommit());
		} else {
			System.out.println("Build info:" 
					+ "Implementation-Build = " + buildVer + ", " 
					+ "Implementation-Version = " + implVer);
		}
		normalShutdown = true;
		// Ensure that we exit even if there are non-daemon threads hanging around
		System.err.println("Finished Exiting...");
		System.exit(0);
	}

	protected DOLDAClassifier getSamplingClassifier(DOLDAConfiguration config,
			DOLDADataSet trainingSetData) {
		// new DOLDASamplingClassifier(config, trainingSetData);
		return new DOLDAPointClassifier(config, trainingSetData);
	}
}
