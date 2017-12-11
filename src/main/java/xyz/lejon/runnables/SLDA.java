package xyz.lejon.runnables;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import org.apache.commons.configuration.ConfigurationException;

import cc.mallet.configuration.LDAConfiguration;
import cc.mallet.topics.LDAGibbsSampler;
import cc.mallet.topics.SpaliasUncollapsedParallelLDA;
import cc.mallet.topics.TopicModelDiagnosticsPlain;
import cc.mallet.types.Alphabet;
import cc.mallet.types.InstanceList;
import cc.mallet.util.LDAUtils;
import joinery.DataFrame;
import xyz.lejon.bayes.models.dolda.DOLDA;
import xyz.lejon.bayes.models.dolda.DOLDADataSet;
import xyz.lejon.bayes.models.dolda.DOLDAEvaluation;
import xyz.lejon.bayes.models.dolda.ModelFactory;
import xyz.lejon.configuration.ConfigFactory;
import xyz.lejon.configuration.Configuration;
import xyz.lejon.configuration.DOLDACommandLineParser;
import xyz.lejon.configuration.DOLDAConfiguration;
import xyz.lejon.configuration.DOLDAPlainLDAConfiguration;
import xyz.lejon.configuration.ParsedDOLDAConfiguration;
import xyz.lejon.eval.EvalResult;
import xyz.lejon.utils.EclipseDetector;
import xyz.lejon.utils.LoggingUtils;
import xyz.lejon.utils.MatrixOps;
import xyz.lejon.utils.Timer;

public class SLDA {
	static DOLDA dolda;
	
	public static String PROGRAM_NAME = "DOProbitSLDA";
	public static PrintWriter pw;
	protected static volatile boolean abort = false;
	protected static volatile boolean normalShutdown = false;
	
	private static DOLDA getCurrentSampler() {
		return dolda;
	}

	public static void main(String[] args) throws Exception {
		
		if(args.length == 0) {
			System.out.println("\n" + PROGRAM_NAME + ": No args given, you should typically call it along the lines of: \n" 
					+ "java -cp DOLDA-X.X.X.jar xyz.lejon.runnables.SLDA --run_cfg=src/main/resources/configuration/DOLDAConfig.cfg\n" 
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
						System.err.println("Running shutdown hook: SLDA Aborted! Waiting for shutdown to finish...");
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

		Thread.setDefaultUncaughtExceptionHandler(new Thread.UncaughtExceptionHandler() {
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
		
		DOLDACommandLineParser cp = new DOLDACommandLineParser(args);
		
		// We have to create this temporary config because at this stage if we want to create a new config for each run
		ParsedDOLDAConfiguration tmpconfig = (ParsedDOLDAConfiguration) ConfigFactory.getMainConfiguration(cp);			
		
		int numberOfRuns = tmpconfig.getInt("no_runs");
		System.out.println("Doing: " + numberOfRuns + " runs");
		// Reading in command line parameters		
		for (int run = 0; run < numberOfRuns && !abort; run++) {
			System.out.println("Starting run: " + run);
			
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
			String [] configs = config.getSubConfigs();
			for(String conf : configs) {
				if(abort) break;
				lu.checkCreateAndSetSubLogDir(conf);
				config.activateSubconfig(conf);
				LoggingUtils.doInitialLogging(cp, (Configuration) config, SLDA.class.getName(), "initial_log", null, lu.getLogDir().getAbsolutePath());
				
				File lgDir = lu.getLogDir();
				
				pw = new PrintWriter(new File(lgDir.getAbsolutePath() + "/" + PROGRAM_NAME + "-crash.txt"));

				System.out.println("Using Config: " + config.whereAmI());
				System.out.println("Runnin subconfig: " + conf);
				String dataset_fn = config.getTextDatasetTrainFilename();
				System.out.println("Using dataset: " + dataset_fn);

				DOLDADataSet trainingSetData =config.loadCombinedTrainingSet();
				DOLDADataSet testSetData = config.loadCombinedTestSet();
				
				// If we have a test dataset, ensure that it is aligned (i.e has a subset of the trainingset labels)
				if(!(testSetData==null || testSetData.isEmpty()) && !DOLDADataSet.ensureAligned(trainingSetData,testSetData)) {
					String [] trainingLabels = trainingSetData.getLabels();
					Set<String> trainLblSet = new TreeSet<>(Arrays.asList(trainingLabels));
					
					String [] testLabels = testSetData.getLabels();
					Set<String> testLblSet = new TreeSet<>(Arrays.asList(testLabels));	
					
					testLblSet.removeAll(trainLblSet);
					
					throw new IllegalStateException("Test set and training sets are not aligned, cannot continue! Ensure that the names and number of class labels are the same in both. \nTraining: " 
							+ trainLblSet + "\nIn Test but not train: " + testLblSet);
				}
				
				double [][] xs = trainingSetData.getX();
				int [] ys = trainingSetData.getY();
				Map<Integer,String> idMap = config.getIdMap();

				dolda = ModelFactory.get(config, xs, ys);
				
				System.out.println("X is: " + MatrixOps.doubleArrayToPrintString(xs, 5));
				
				dolda.setRandomSeed(commonSeed);
				System.out.println(String.format("Rare word threshold: %d", config.getRareThreshold(DOLDAConfiguration.RARE_WORD_THRESHOLD)));

				InstanceList textData = trainingSetData.getTextData();
				if(textData!=null) {
					System.out.println("Vocabulary size: " + textData.getDataAlphabet().size() + "\n");
					System.out.println("Instance list is: " + textData.size());
					System.out.println("Loading data instances...");
					textData.getAlphabet().stopGrowth();
				}
				// Sets the frequent with which top words for each topic are printed
				//model.setShowTopicsInterval(config.getTopicInterval(DOLDAConfiguration.TOPIC_INTER_DEFAULT));
				dolda.setRandomSeed(config.getSeed(DOLDAConfiguration.SEED_DEFAULT));
				System.out.println("Config seed:" + config.getSeed(DOLDAConfiguration.SEED_DEFAULT));
				System.out.println("Start seed: " + dolda.getStartSeed());
				// Imports the data into the model
				if(textData!=null) {
					dolda.addInstances(textData);
				}
				System.out.println("Starting iterations (" + config.getNoIterations(DOLDAConfiguration.NO_ITER_DEFAULT) + " total).");
				System.out.println("_____________________________\n");

				// Runs the model
				System.out.println("Starting:" + new Date());
				Timer t = new Timer();
				t.start();
				dolda.sample(config.getNoIterations(DOLDAConfiguration.NO_ITER_DEFAULT));
				t.stop();
				System.out.println("Finished:" + new Date());
				
				/* ========= Evaluation stage =============== */ 
				double [][] betas = dolda.getBetas();	
				
				double [][] testset;
				int [] testLabels;
				
				if(testSetData==null || testSetData.isEmpty() ) {
					double[][] zbar_d = dolda.getSupervsedTopicIndicatorMeans();
					testset = MatrixOps.concatenate(xs,zbar_d);
					testLabels = ys;
				} else {
					double [][] testZ = null;
					if(testSetData.getTextData()!=null && testSetData.getTextData().size()>0) {
						// Here we sample the topic indicators using ordinary LDA 
						// with the Phi we have learned during sampling!
						Alphabet trainingAlphabet = textData != null ? textData.getAlphabet() : null;
						testZ = sampleTestTopicIndicatorsMeans(config, commonSeed, dolda.getPhiMeans(),trainingAlphabet);
						
						if(config.saveDocumentTopicMeans()) {
							// Save the  doc-topics-means of the testset too 
							String testDocTopicMeansOutFullFn = config.getDocumentTopicMeansOutputFilename();
							File fn = new File(testDocTopicMeansOutFullFn);
							String testDocTopicMeansOutFn = "TESTSET-" + fn.getName();
							ExperimentUtils.saveDocTopicMeans(lgDir, testZ, testDocTopicMeansOutFn);
						}
					} else {
						testZ = new double [testSetData.getX().length][0];
					}
					
					double [][] additionalCovariates = testSetData.getX();
					if(additionalCovariates==null) {
						// even if additionalCovariates is null, teztZ MUST NOT be null
						additionalCovariates = new double [testZ.length][0];
					}
					
					testset = MatrixOps.concatenate(additionalCovariates,testZ);
					testLabels = testSetData.getY();
				}

				DOLDAEvaluation eval = new DOLDAEvaluation();
				EvalResult result  = eval.evaluate(testset, testLabels, betas);
				double pCorrect = ((double)result.noCorrect)/testset.length * 100;
				
				if(textData!=null) {
					double[][] zbar_d = dolda.getSupervsedTopicIndicatorMeans();
					for (int j = 0; j < zbar_d.length; j++) {
						System.out.print("Doc " + j + "=" + textData.get(j).getTarget().toString() + ": ");
						for (int j2 = 0; j2 < zbar_d[j].length; j2++) {
							System.out.print("[" + j2 + " => " + MatrixOps.formatDouble(zbar_d[j][j2]) + "], ");
						}
						System.out.println();
					}
				}
				
				System.out.println("Final Betas: " + MatrixOps.doubleArrayToPrintString(betas));
				System.out.println();
				System.out.println("Confusion Matrix: \n" + DOLDAEvaluation.confusionMatrixToString(result.confusionMatrix,idMap));
				System.out.println();
				System.out.println("Total correct: " + result.noCorrect + " / " + ((double)testset.length) +  " => " + String.format("%.0f",pCorrect) + "% correct");
				System.out.println(PROGRAM_NAME + " took: " + (t.getEllapsedTime() / 1000) + " seconds");
				
				String [] xColnames = trainingSetData.getColnamesX();
				if(xColnames==null) { xColnames = new String[0]; }
				Integer noSupervisedTopics = config.getNoSupervisedTopics(0);
				if(textData==null) {
					noSupervisedTopics = 0;
				}
				
				//String [] allColnames = new String[xColnames.length+noSupervisedTopics];
				String [] allColnames = ExperimentUtils.createColumnLabelsFromNames(xColnames, trainingSetData.getColnameY(), noSupervisedTopics);
				
				// Save last betas
				if(config.getSaveBetas()) {
					ExperimentUtils.saveBetas(lgDir, allColnames, xs[0].length, betas, config.getIdMap(), config.betasOutputFn());
				}
				
				// Save sampled betas if that is turned on in config
				if(config.saveBetaSamples()) {
					List<double []> [] sampledBetas = dolda.getSampledBetas();
					ExperimentUtils.saveBetaSamples(lgDir, allColnames, xs[0].length, sampledBetas, config.getIdMap(), config.betaSamplesOutputFn());
				}
				
				if(config.saveDocumentTopicMeans() 
						&& config.getTextDatasetTrainFilename()!=null 
						&& config.getTextDatasetTrainFilename().trim().length()>0) {
					ExperimentUtils.saveDocTopicMeans(lgDir, dolda.getZbar(), config.getDocumentTopicMeansOutputFilename());
				}
				
				
				if(config.savePhiMeans(LDAConfiguration.SAVE_PHI_MEAN_DEFAULT)) {
					String phiMeanFn = config.getPhiMeansOutputFilename();
					double [][] means = dolda.getPhiMeans();
					if(means!=null) {
						LDAUtils.writeASCIIDoubleMatrix(means, lgDir.getAbsolutePath() + "/" + phiMeanFn, ",");
					} else {
						System.err.println("WARNING: ParallelLDA: No Phi means where sampled, not saving Phi means! This is likely due to a combination of configuration settings of phi_mean_burnin, phi_mean_thin and save_phi_mean");
					}
					// No big point in saving Phi without the vocabulary
					String vocabFn = config.getVocabularyFilename();
					if(vocabFn==null || vocabFn.length()==0) { vocabFn = "phi_vocabulary.txt"; }
					String [] vobaculary = LDAUtils.extractVocabulaty(textData.getDataAlphabet());
					LDAUtils.writeStringArray(vobaculary,lgDir.getAbsolutePath() + "/" + vocabFn);
				}
				
				// Save document topic diagnostics
				if(config.saveDocumentTopicDiagnostics()) {
					if(dolda instanceof LDAGibbsSampler) {			
						LDAGibbsSampler model = (LDAGibbsSampler) dolda;
						int requestedWords = config.getNrTopWords(LDAConfiguration.NO_TOP_WORDS_DEFAULT);
						TopicModelDiagnosticsPlain tmd = new TopicModelDiagnosticsPlain(model, requestedWords);
						System.out.println("Topic model diagnostics:");
						System.out.println(tmd.toString());			
						String docTopicDiagFn = config.getDocumentTopicDiagnosticsOutputFilename();
						PrintWriter out = new PrintWriter(lgDir.getAbsolutePath() + "/" + docTopicDiagFn);
						out.println(tmd.topicsToCsv());
						out.flush();
						out.close();
					} else {
						throw new RuntimeException("Sampler is not an instance of an LDAGibbsSampler, cannot exctract statistics");
					}
				}

				if(config.saveVocabulary(false)) {
					String vocabFn = config.getVocabularyFilename();
					String [] vobaculary = LDAUtils.extractVocabulaty(textData.getDataAlphabet());
					LDAUtils.writeStringArray(vobaculary,lgDir.getAbsolutePath() + "/" + vocabFn);
				}
				
				if(config.saveTermFrequencies(false)) {
					String termCntFn = config.getTermFrequencyFilename();
					int [] freqs = LDAUtils.extractTermCounts(textData);
					LDAUtils.writeIntArray(freqs, lgDir.getAbsolutePath() + "/" + termCntFn);
				}
				
				if(config.saveDocLengths(false)) {
					String docLensFn = config.getDocLengthsFilename();
					int [] freqs = LDAUtils.extractDocLength(textData);
					LDAUtils.writeIntArray(freqs, lgDir.getAbsolutePath() + "/" + docLensFn);
				}
				
				/*
				if(config.doPlot()) {
					ClassificationResultPlot.plot2D(labels, xs);					
				}
				*/

				List<String> metadata = new ArrayList<String>();
				if(dolda.getAbort()) {
					metadata.add("!!!! WARNING !!!! sampling was aborted. Iterations sampled: " + dolda.getCurrentIteration());
				}
				metadata.add("No. Topics: " + dolda.getNoTopics());
				metadata.add("No. Supervised Topics: " + config.getNoSupervisedTopics(-1));
				metadata.add("Start Seed: " + dolda.getStartSeed());
				metadata.add("Accuracy: " + String.format("%.0f",pCorrect));
				metadata.add("ConfusionMatrix: " + "\n" + DOLDAEvaluation.confusionMatrixToString(result.confusionMatrix,idMap));
				// Save stats for this run
				lu.dynamicLogRun("Runs", t, cp, (Configuration) config, null, 
						SLDA.class.getName(), conf+"-results", "HEADING", "DOLDA", numberOfRuns, metadata);
				
				boolean fakeTextData = trainingSetData.hasFakeTextData();
				int requestedWords = config.getNrTopWords(LDAConfiguration.NO_TOP_WORDS_DEFAULT);
				if(!fakeTextData && requestedWords>dolda.getAlphabet().size()) {
					requestedWords = dolda.getAlphabet().size();
				}

				PrintWriter topOut = new PrintWriter(lgDir.getAbsolutePath() + "/TopWords.txt");
				if(!fakeTextData) {
					String topWords = LDAUtils.formatTopWords(
							LDAUtils.getTopWords(requestedWords, 
									dolda.getAlphabet().size(), 
									dolda.getNoTopics(), 
									dolda.getTypeTopicMatrix(), 
									dolda.getAlphabet()));
					System.out.println("Top words are: \n" + topWords);
					topOut.println(topWords);
					topOut.flush();
					topOut.close();
					topOut = new PrintWriter(lgDir.getAbsolutePath() + "/TopWords.csv");
					topWords = LDAUtils.formatTopWordsAsCsv(
							LDAUtils.getTopWords(requestedWords, 
									dolda.getAlphabet().size(), 
									dolda.getNoTopics(), 
									dolda.getTypeTopicMatrix(), 
									dolda.getAlphabet()));
					topOut.println(topWords);
					
					PrintWriter out = new PrintWriter(lgDir.getAbsolutePath() + "/RelevanceWords.txt");
					topWords = LDAUtils.formatTopWords(
							LDAUtils.getTopRelevanceWords(requestedWords, 
									dolda.getAlphabet().size(), 
									dolda.getNoTopics(), 
									dolda.getTypeTopicMatrix(),  
									config.getBeta(LDAConfiguration.BETA_DEFAULT),
									config.getLambda(LDAConfiguration.LAMBDA_DEFAULT), 
									dolda.getAlphabet()));
					System.out.println("Relevance words are: \n" + topWords);
					out.println(topWords);
					out.flush();
					out.close();
					out = new PrintWriter(lgDir.getAbsolutePath() + "/RelevanceWords.csv");
					topWords = LDAUtils.formatTopWordsAsCsv(
							LDAUtils.getTopRelevanceWords(requestedWords, 
									dolda.getAlphabet().size(), 
									dolda.getNoTopics(), 
									dolda.getTypeTopicMatrix(),  
									config.getBeta(LDAConfiguration.BETA_DEFAULT),
									config.getLambda(LDAConfiguration.LAMBDA_DEFAULT), 
									dolda.getAlphabet()));
					out.println(topWords);
					out.flush();
					out.close();
				} else { 
					topOut.println("No text data used");
				}
				topOut.flush();
				topOut.close();

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
	}

	static void saveBetas(DOLDAConfiguration config, File lgDir, DOLDADataSet trainingSetData, double[][] xs,
			Map<Integer, String> idMap, double[][] betas) throws IOException {
		String [] columnLabels = new String[betas[0].length+1];
		columnLabels[0] = "Class";
		for (int lblIdx = 1; lblIdx < columnLabels.length; lblIdx++) {
			String [] xColnames = trainingSetData.getColnamesX();
			// In the output name colums Xn for X covariates and Zn for supervised topics
			// We have <= since we have added the "Class" column
			if(lblIdx <= xs[0].length) {
				columnLabels[lblIdx] = xColnames[lblIdx-1];
			} else {
				columnLabels[lblIdx] = "Z" + (lblIdx-xs[0].length);
			}
		}
		DataFrame<Object> out = new DataFrame<>(columnLabels);
		for (int j = 0; j < betas.length; j++) {
			List<Object> row = new ArrayList<>();
			//Add the class id to the first column
			row.add(idMap.get(j));
			for (int k = 0; k < betas[j].length; k++) {
				row.add(betas[j][k]);
			}
			out.append(row);
		}
		out.writeCsv(lgDir.getAbsolutePath() + "/" + config.betasOutputFn());
		
		
		if(config.saveBetaSamples()) {
			out = new DataFrame<>(columnLabels);
			List<double []> [] sampledBetas = dolda.getSampledBetas();
			for (int k = 0; k < sampledBetas.length; k++) {
				List<double []> betasClassK = sampledBetas[k];
				for (int j = 0; j < betasClassK.size(); j++) {
					double [] betaRow = betasClassK.get(j);
					List<Object> row = new ArrayList<>();
					//Add the class id to the first column
					row.add(idMap.get(k));
					for (int covariate = 0; covariate < betaRow.length; covariate++) {
						row.add(betaRow[covariate]);
					}
					out.append(row);
				}
			}
			if(out.length()>0) out.writeCsv(lgDir.getAbsolutePath() + "/" + config.betaSamplesOutputFn());
		}
	}

	public static double[][] sampleTestTopicIndicatorsMeans(DOLDAConfiguration config, int commonSeed, double [][] phi, Alphabet trainingAlphabet) throws ConfigurationException,
			FileNotFoundException, IOException {
		SpaliasUncollapsedParallelLDA model = new SpaliasUncollapsedParallelLDA(new DOLDAPlainLDAConfiguration(config));
		
		String test_dataset_fn = config.getTextDatasetTestFilename();
		System.out.println("Using test dataset: " + test_dataset_fn);

		InstanceList instances = LDAUtils.loadDataset(new DOLDAPlainLDAConfiguration(config), test_dataset_fn, trainingAlphabet);
		//InstanceList instances = LDAUtils.loadInstancesPrune(test_dataset_fn, 
		//		config.getStoplistFilename("stoplist.txt"), config.getRareThreshold(LDAConfiguration.RARE_WORD_THRESHOLD), config.keepNumbers());
		
		model.setRandomSeed(commonSeed);
		System.out.println("Start seed: " + model.getStartSeed());
		System.out.println(String.format("Rare word threshold: %d", config.getRareThreshold(LDAConfiguration.RARE_WORD_THRESHOLD)));

		System.out.println("Testset Vocabulary size: " + instances.getDataAlphabet().size() + "\n");
		System.out.println("No. Test instances: " + instances.size());
		System.out.println("Loading test instances...");

		// Imports the data into the model
		model.addInstances(instances);

		System.out.println("Starting sampling the test data (" + config.getNoIterations(LDAConfiguration.NO_ITER_DEFAULT) + " iterations).");
		System.out.println("_____________________________\n");

		// Runs the model
		model.setPhi(MatrixOps.clone(phi), instances.getAlphabet(), instances.getTargetAlphabet());
		model.sampleZGivenPhi(config.getNoIterations(LDAConfiguration.NO_ITER_DEFAULT));
		
		System.out.println("Test sampling finished.");
		
		return MatrixOps.extractCols(0,config.getNoSupervisedTopics(-1), model.getZbar());
	}

}
