package xyz.lejon.bayes.models.dolda;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.apache.commons.configuration.ConfigurationException;

import xyz.lejon.configuration.DOLDAConfigUtils;
import xyz.lejon.configuration.DOLDAConfiguration;
import xyz.lejon.configuration.DOLDAPlainLDAConfiguration;
import xyz.lejon.utils.EnhancedConfusionMatrix;
import xyz.lejon.utils.MatrixOps;
import xyz.lejon.utils.Timer;
import cc.mallet.classify.Classification;
import cc.mallet.classify.Trial;
import cc.mallet.topics.SpaliasUncollapsedParallelLDA;
import cc.mallet.types.CrossValidationIterator;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelVector;

/**
 * This class uses a callback mechanism to hook into the running 
 * sampler and can thus sample also the predictions as to get a
 * sampling distribution for the predictions 
 * 
 * @author Leif Jonsson
 *
 */
public class DOLDASamplingClassifier extends DOLDAPointClassifier implements DOLDAClassifier, DOLDAIterationCallback {

	private static final long serialVersionUID = 1L;
	protected InstanceList currentTestset;
	protected int iterationsDone = 0;
	protected int burn_in = 0;
	protected int lag = 1;
	Map<Instance, double []> instanceToPredictiveDistribution = new HashMap<Instance, double[]>();

	public DOLDASamplingClassifier(DOLDAConfiguration config, DOLDADataSet dataset) {
		super(config,dataset);
		int burnIn = config.getBurnIn();
		Integer noIterations = config.getNoIterations(DOLDAConfiguration.NO_ITER_DEFAULT);
		burn_in = (int) (((double) burnIn / 100.0) * ((double) noIterations));
		lag = config.getLag();
		spaliasIterations = 100;
	}
	
	public Map<Instance, double []> getPredictiveDistributions() {
		return instanceToPredictiveDistribution;
	}
	
	/**
	 * In this overridden version, we simply select the class 
	 * which has most predictions the MAP estimate
	 *  
	 * @see xyz.lejon.bayes.models.dolda.DOLDAPointClassifier#classify(cc.mallet.types.Instance)
	 */
	@Override
	public Classification classify(Instance instance) {
		double [] scores = instanceToPredictiveDistribution.get(instance).clone();
		
		// scores contains the absolute number of times each category has been classified
		// so we need to normalize the scores to get a distribution for it
		double sum = MatrixOps.sum(scores);
		for (int i = 0; i < scores.length; i++) {
			scores[i] /= sum;
		}
		
		return new Classification(instance, this, new LabelVector (getLabelAlphabet(), scores));
	}
	
	@Override
	protected void cvIncText(InstanceList instances, int folds, Trial [] trials) throws Exception {
		Random r = new Random ();
		int TRAINING = 0;
		int TESTING = 1;
		CrossValidationIterator cvIter = getCrossValidationIterator(instances, folds, r);
		InstanceList[] cvSplit = null;

		for (int fold = 0; fold < folds && !abort; fold++) {
			iterationsDone  = 0;
			cvSplit = cvIter.next();
			trainXs = DOLDAConfigUtils.extractXs(cvSplit[TRAINING], fullDataset.getX(), fullDataset.getRowIds());
			trainYs = DOLDAConfigUtils.extractYs(cvSplit[TRAINING], fullDataset.getY(), fullDataset.getRowIds());
			trainRowIds = DOLDAConfigUtils.extractRowIds(cvSplit[TRAINING], fullDataset.getX(), fullDataset.getRowIds());

			testXs = DOLDAConfigUtils.extractXs(cvSplit[TESTING], fullDataset.getX(), fullDataset.getRowIds());
			testYs = DOLDAConfigUtils.extractYs(cvSplit[TESTING], fullDataset.getY(), fullDataset.getRowIds());
			testRowIds = DOLDAConfigUtils.extractRowIds(cvSplit[TESTING], fullDataset.getX(), fullDataset.getRowIds());

			int tries = 0;
			int maxTries = 3;
			boolean success = false;
			DOLDAWithCallback dolda = null;
			Exception trainingException = null;
			// Sometimes the gamma or lambda sampling returns NaN and the beta sampling aborts
			// let's try a couple of times before giving up so perhaps we can save 
			// a couple of cross validations
			while(!success && tries < maxTries && !abort) {
				try {
					dolda = initiateSampler( cvSplit[TRAINING] );
					lastDolda = dolda;
					currentTestset = cvSplit[TESTING];
					
					System.out.println("Starting iterations (" + config.getNoIterations(DOLDAConfiguration.NO_ITER_DEFAULT) + " total).");
					System.out.println("_____________________________\n");

					System.out.println("Starting:" + new Date());
					Timer t = new Timer();
					t.start();
					dolda.sample(config.getNoIterations(DOLDAConfiguration.NO_ITER_DEFAULT));
					t.stop();
					System.out.println("Finished:" + new Date());
					
					success = true;	    			
				} catch (Exception e1) {
					System.err.println("Training failed: " + e1);
					System.err.println("Retrying (" + tries + "/" + maxTries + ")...");
					trainingException = e1;
					tries++;
				}
			}
			if(!success) {
				System.err.println("Training failed, giving up after " + tries + " tries...");
				throw trainingException;
			}

			trials[fold] = new Trial(this, cvSplit[TESTING]);
			System.out.println("Trial accuracy: "  + trials[fold].getAccuracy());
			EnhancedConfusionMatrix enhancedConfusionMatrix = new EnhancedConfusionMatrix(trials[fold]);
			System.out.println("Trial confusion matrix: \n"  + enhancedConfusionMatrix);
			
			if(instanceToPredictiveDistribution!=null && instanceToPredictiveDistribution.size() > 0) {
				PrintWriter out = new PrintWriter(config.getLoggingUtil().getLogDir().getAbsolutePath() + "/predictive-distribution-fold-" + fold + ".csv");
				out.println(predictiveDistributionToCsvHeaders());
				out.println(predictiveDistributionToCsv());
				out.flush();
				out.close();
				instanceToPredictiveDistribution.clear();
			}
			saveFoldData(fold, enhancedConfusionMatrix, dolda);
		}
	}
	
	public DOLDAWithCallback initiateSampler(InstanceList trainingSet) throws IOException {		
		DOLDAWithCallback dolda = ModelFactory.get(config, trainXs, trainYs, this);

		//System.out.println("X is: " + MatrixOps.doubleArrayToPrintString(trainXs, 5));
		int commonSeed = config.getSeed(DOLDAConfiguration.SEED_DEFAULT);

		dolda.setRandomSeed(commonSeed);
		System.out.println(String.format("Rare word threshold: %d", config.getRareThreshold(DOLDAConfiguration.RARE_WORD_THRESHOLD)));

		if(trainingSet!=null) {
			System.out.println("Vocabulary size: " + trainingSet.getDataAlphabet().size() + "\n");
			System.out.println("Instance list is: " + trainingSet.size());
			System.out.println("Loading data instances...");
		}
		// Sets the frequent with which top words for each topic are printed
		//model.setShowTopicsInterval(config.getTopicInterval(DOLDAConfiguration.TOPIC_INTER_DEFAULT));
		dolda.setRandomSeed(config.getSeed(DOLDAConfiguration.SEED_DEFAULT));
		System.out.println("Config seed:" + config.getSeed(DOLDAConfiguration.SEED_DEFAULT));
		System.out.println("Start seed: " + dolda.getStartSeed());
		// Imports the data into the model
		if(trainingSet!=null && !fakeTextData ) {
			dolda.addInstances(trainingSet);
		}
		return dolda;
	}

	@Override
	public void iterationState(DOLDAIterationState state) {
		if(iterationsDone > burn_in && (iterationsDone % lag) == 0) {
			double [][] phi = state.getPhi();
			double [][] betas = state.getBetas();

			if(!fakeTextData) {
				SpaliasUncollapsedParallelLDA spalias;
				try {
					spalias = new SpaliasUncollapsedParallelLDA(new DOLDAPlainLDAConfiguration(config));
				} catch (ConfigurationException e) {
					e.printStackTrace();
					throw new IllegalArgumentException("DOLDASamplingClassifier: not a valid configuration");
				}
				spalias.addInstances(currentTestset);
				spalias.setPhi(MatrixOps.clone(phi), getAlphabet(), getLabelAlphabet());
				spalias.sampleZGivenPhi(spaliasIterations);
				// sampledZbar will contain the total number of topics
				double [][] sampledTestZBar = spalias.getZbar();
				// Pick out the 'supervised' topics...
				sampledSupervisedTestTopics = MatrixOps.extractCols(0,config.getNoSupervisedTopics(-1), sampledTestZBar);
			}
			else {
				sampledSupervisedTestTopics = new double[testXs.length][];
				for (int i = 0; i < sampledSupervisedTestTopics.length; i++) {
					sampledSupervisedTestTopics[i] = new double[0];
				}
			}
			classifyTestset(betas);
		}
		
		iterationsDone++;
	}

	protected void classifyTestset(double[][] betas) {
		noCorrect = 0;
		for(Instance instance : currentTestset) {
			classifyInstance(instance, betas);
		}
		//System.out.println("No correct: " + noCorrect);
	}

	/**
	 * This method is similar to the classify method, the difference
	 * is that this method will be called during sampling (not after)
	 * with one sampled <code>beta</code> instance. It saves the 
	 * prediction to the <code>instanceToPredictiveDistribution</code>
	 * Map, which is used in the end to do the final classification 
	 * of the instance.
	 * 
	 * @param instance
	 * @param betas
	 */
	private void classifyInstance(Instance instance, double[][] betas) {
		String instanceId = instance.getName().toString();

		int row = findXrow(instanceId, testRowIds);

		int [] ys = testYs;

		int predClass;
		int realClass;

		double [][] xrow = new double[1][testXs[row].length+sampledSupervisedTestTopics[row].length];
		xrow[0] = MatrixOps.concatenate(testXs[row],sampledSupervisedTestTopics[row]);

		double[] scores = calcClassScores(xrow, betas);
		int categoryIdx = MatrixOps.maxIdx(scores);
				
		predClass = categoryIdx;
		realClass = ys[row];
		if(predClass==realClass) {
			if(verbose) System.out.println(row + ": True: " + realClass + "\t Predicted: " + predClass + "\t => Correct!");
			noCorrect++;
		} else {
			if(verbose) System.out.println(row + ": True: " + realClass + "\t Predicted: " + predClass + "\t => Incorrect!");
		}

		//Classification classification = new Classification(instance, this, new LabelVector (getLabelAlphabet(), scores));
		if(instanceToPredictiveDistribution.get(instance)==null) {
			instanceToPredictiveDistribution.put(instance, new double [scores.length]);
		}
		// Increase the category @ categoryIdx with one, since we have one more prediction for it
		instanceToPredictiveDistribution.get(instance)[categoryIdx]++;
	}
	
	public String predictiveDistributionToString() {
		String result = "";
		for (Instance instance : instanceToPredictiveDistribution.keySet()) {
			result += instance.getName().toString() + " => " + MatrixOps.arrToStr(instanceToPredictiveDistribution.get(instance)) + "\n";
		}
		return result;
	}

	public String predictiveDistributionToCsv() {
		String result = "";
		for (Instance instance : instanceToPredictiveDistribution.keySet()) {
			result += instance.getName().toString() + "," + MatrixOps.arrToCsv(instanceToPredictiveDistribution.get(instance)) + "\n";
		}
		return result;
	}

	public String predictiveDistributionToCsvHeaders() {
		String result = "Id,";
		Map<Integer,String> idToLabel = fullDataset.getIdToLabels();
		int categoryCnt = idToLabel.size();
		for (int i = 0; i < categoryCnt; i++) {
			result += idToLabel.get(i);
			if(i<(categoryCnt-1)) result += ",";
		}
		return result;
	}
}
