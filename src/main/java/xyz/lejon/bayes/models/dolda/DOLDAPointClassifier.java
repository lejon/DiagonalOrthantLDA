package xyz.lejon.bayes.models.dolda;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Date;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import org.apache.commons.configuration.ConfigurationException;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import xyz.lejon.configuration.DOLDAConfigUtils;
import xyz.lejon.configuration.DOLDAConfiguration;
import xyz.lejon.configuration.DOLDAPlainLDAConfiguration;
import xyz.lejon.runnables.ExperimentUtils;
import xyz.lejon.utils.EnhancedConfusionMatrix;
import xyz.lejon.utils.MatrixOps;
import xyz.lejon.utils.Timer;
import cc.mallet.classify.Classification;
import cc.mallet.classify.Classifier;
import cc.mallet.classify.Trial;
import cc.mallet.configuration.LDAConfiguration;
import cc.mallet.topics.SpaliasUncollapsedParallelLDA;
import cc.mallet.types.Alphabet;
import cc.mallet.types.AlphabetCarrying;
import cc.mallet.types.CrossValidationIterator;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelAlphabet;
import cc.mallet.types.LabelVector;
import cc.mallet.util.LDAUtils;
import cc.mallet.util.Randoms;

/**
 * This class is really just a big hack trying to get around the fact that MALLET
 * is built around handling text data. 
 * 
 * One 'notable' feature is that in case of no text data, it will create a fake
 * text dataset so we can do cross validation the same way no matter if we have 
 * text covariates or not
 * 
 * @author Leif Jonsson
 *
 */
public class DOLDAPointClassifier extends Classifier implements DOLDAClassifier {

	protected static int spaliasIterations = 1000;

	private static final long serialVersionUID = 1L;

	DOLDAConfiguration config;
	boolean verbose = false;
	double [][] betas;
	List<double []> [] sampledBetas;
	DOLDA lastDolda;
	DOLDADataSet fullDataset;
	int noCorrect = 0;

	double [][] trainXs;
	int [] trainYs;

	double [][] testXs;
	int [] testYs;
	String [] testRowIds;
	String [] trainRowIds;

	double [][] sampledSupervisedTestTopics;

	boolean fakeTextData = false;

	boolean abort = false;
	
	List<String> fixedTrainingIds = null;

	public DOLDAPointClassifier(DOLDAConfiguration config, DOLDADataSet dataset) {
		this.config = config;
		this.fullDataset = dataset;
		fakeTextData = dataset.hasFakeTextData();
		if(dataset.textData!=null) {
			instancePipe = dataset.textData.getPipe();
		}
		spaliasIterations = config.getNoTestIterations(DOLDAConfiguration.NO_ITER_DEFAULT);
	}
	
	// TODO: This code should be improved to handle instances from both train, test and validation
	// sets. It is also inefficient in concatenation the WHOLE textXs and sampledSupervisedTestTopics
	// when we only need one row!
	/* (non-Javadoc)
	 * @see xyz.lejon.bayes.models.dolda.DOLDAClassifier#classify(cc.mallet.types.Instance)
	 */
	@Override
	public Classification classify(Instance instance) {
		String instanceId = instance.getName().toString();

		int row = findXrow(instanceId, testRowIds);

		double [][] xs = MatrixOps.concatenate(testXs,sampledSupervisedTestTopics);
		int [] ys = testYs;

		int predClass;
		int realClass;

		double [][] xrow = new double[1][xs[row].length];
		xrow[0] = xs[row];

		double[] scores = calcClassScores(xrow, betas);
		int maxIdx = MatrixOps.maxIdx(scores);
		
		predClass = maxIdx;
		realClass = ys[row];
		if(predClass==realClass) {
			if(verbose) System.out.println(row + ": True: " + realClass + " Predicted: " + predClass + " => Correct!");
			noCorrect++;
		} else {
			if(verbose) System.out.println(row + ": True: " + realClass + " Predicted: " + predClass + " => Incorrect!");
		}

		return new Classification(instance, this, new LabelVector (getLabelAlphabet(), scores));
	}

	protected double[] calcClassScores(double[][] xrow, double [][] betas) {
		DenseMatrix64F xrowd = new DenseMatrix64F(xrow);
		DenseMatrix64F Betas = new DenseMatrix64F(betas);
		DenseMatrix64F aHatd = new DenseMatrix64F(xrowd.numRows,Betas.numRows);
		CommonOps.multTransB(xrowd, Betas, aHatd);
		double [] scores = MatrixOps.extractDoubleVector(aHatd);
		return scores;
	}

	@Override
	public LabelAlphabet getLabelAlphabet() {
		return super.getLabelAlphabet();
	}

	protected int findXrow(String instanceId, String [] rowIds) {
		int idx = 0;
		for(String iid : rowIds) {
			if(iid.equals(instanceId)) return idx;
			idx++;
		}
		throw new IllegalArgumentException("Could not find id: " + instanceId + " in dataset!");
	}

	public DOLDA train(InstanceList trainingSet) throws IOException {		
		DOLDA dolda = ModelFactory.get(config, trainXs, trainYs);

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
		betas = dolda.getBetas();	
		sampledBetas = dolda.getSampledBetas();
		lastDolda = dolda;
		return dolda;
	}

	class SimpleStringData implements AlphabetCarrying {
		Alphabet alphabet;
		String data;

		public SimpleStringData(Alphabet alphabet, String data) {
			super();
			this.alphabet = alphabet;
			this.data = data;
		}

		@Override
		public Alphabet getAlphabet() {
			return alphabet;
		}

		@Override
		public Alphabet[] getAlphabets() {
			Alphabet [] as = new Alphabet[1];
			as[0] = alphabet;
			return as;
		}

		@Override 
		public String toString() {
			return data;
		}
	}

	class EmptyInstanceIterator implements Iterator<Instance> {
		String [] labels;
		String [] ids;
		LabelAlphabet targetAlphabet;
		Alphabet dataAlphabet;
		int index;

		public EmptyInstanceIterator (String[] labels, String [] ids, LabelAlphabet targetAlphabet, Alphabet alphabet)
		{
			this.labels = labels;
			this.ids = ids;
			this.targetAlphabet = targetAlphabet;
			this.index = 0;
			dataAlphabet = alphabet;
		}

		public Instance next ()
		{
			String instanceId = ids[index];
			String instanceLabel = labels[index++];
			Instance fakeInstance = new Instance (new SimpleStringData(dataAlphabet, ""), instanceLabel, instanceId, null);
			fakeInstance.setLabeling(targetAlphabet.lookupLabel(instanceLabel));
			return fakeInstance;
		}

		public boolean hasNext ()	{	return index < labels.length;	}

		public void remove () {
			throw new IllegalStateException ("This Iterator<Instance> does not support remove().");
		}
	}

	/* (non-Javadoc)
	 * @see xyz.lejon.bayes.models.dolda.DOLDAClassifier#crossValidate(cc.mallet.types.InstanceList, int)
	 */
	@Override
	public Trial [] crossValidate(InstanceList instances, int folds) throws Exception {
		Trial [] trials = new Trial[folds];
		cvIncText(instances, folds, trials);

		return trials; 
	}

	protected void cvIncText(InstanceList instances, int folds, Trial [] trials) throws Exception {
		Random r = new Random ();
		int TRAINING = 0;
		int TESTING = 1;
		CrossValidationIterator cvIter = getCrossValidationIterator(instances, folds, r);
		InstanceList[] cvSplit = null;

		for (int fold = 0; fold < folds && !abort; fold++) {
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
			DOLDA dolda = null;
			Exception trainingException = null;
			// Sometimes the gamma or lambda sampling returns NaN and the beta sampling aborts
			// let's try a couple of times before giving up so perhaps we can save 
			// a couple of cross validations
			while(!success && tries < maxTries && !abort) {
				try {
					dolda = train( cvSplit[TRAINING] );
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

			if(!fakeTextData) {
				double [][] phi = dolda.getPhi();

				try {
					sampleTestset(TESTING, cvSplit, phi); 

				} catch (ConfigurationException e) {
					throw new IllegalArgumentException(e);
				}
			} else {
				sampledSupervisedTestTopics = new double[testXs.length][];
				for (int i = 0; i < sampledSupervisedTestTopics.length; i++) {
					sampledSupervisedTestTopics[i] = new double[0];
				}
			}

			trials[fold] = new Trial(this, cvSplit[TESTING]);
			System.out.println("Trial accuracy: "  + trials[fold].getAccuracy());
			EnhancedConfusionMatrix enhancedConfusionMatrix = new EnhancedConfusionMatrix(trials[fold]);
			System.out.println("Trial confusion matrix: \n"  + enhancedConfusionMatrix);
			
			saveFoldData(fold, enhancedConfusionMatrix, dolda);
		}
	}

	protected CrossValidationIterator getCrossValidationIterator(InstanceList instances, int folds, Random r) {
		if(getFixedTrainingIds()==null) {			
			return new CrossValidationIterator(instances, folds, r);
		} else {
			return new FixedCrossValidationIterator(instances, folds, r, fixedTrainingIds);
		}
	}

	void saveFoldData(int fold, EnhancedConfusionMatrix enhancedConfusionMatrix, DOLDA dolda) throws IOException {
		String [] xColnames = fullDataset.getColnamesX();
		if(xColnames==null) { xColnames = new String[0]; }
		String [] allColnames = new String[xColnames.length+config.getNoSupervisedTopics(0)];
		for (int i = 0; i < allColnames.length; i++) {
			if(i<xColnames.length) {
				allColnames[i] = xColnames[i];
			} else {
				allColnames[i] = i + "";
			}
		}
		File lgDir = config.getLoggingUtil().getLogDir();
		
		// Save example betas
		String foldPrefix = "fold-";
		if(config.getSaveBetas()) {
			ExperimentUtils.saveBetas(lgDir, allColnames, trainXs[0].length, dolda.getBetas(), config.getIdMap(), foldPrefix + fold + "-" + config.betasOutputFn());
		}
		
		// Save example beta samples if that is turned on in config
		if(config.saveBetaSamples()) {
			ExperimentUtils.saveBetaSamples(lgDir, allColnames, trainXs[0].length, dolda.getSampledBetas(), config.getIdMap(), foldPrefix + fold + "-" + config.betaSamplesOutputFn());
		}
		
		// Save example doc-topic means if that is turned on in config
		if(config.saveDocumentTopicMeans() && config.getTextDatasetTrainFilename()!=null) {
			String dtFn = config.getDocumentTopicMeansOutputFilename();
			ExperimentUtils.saveDocTopicMeans(lgDir, dolda.getZbar(), foldPrefix + fold + "-" + dtFn);
			ExperimentUtils.saveDocTopicMeans(lgDir, sampledSupervisedTestTopics, foldPrefix + fold + "-TESTSET-" + dtFn);
		}
		
		PrintWriter idsOut = new PrintWriter(config.getLoggingUtil().getLogDir().getAbsolutePath() + "/test-ids-fold-" + fold + ".txt");
		for (String id : testRowIds) {				
			idsOut.println(id);
		}
		idsOut.flush();
		idsOut.close();
		
		PrintWriter trainIdsOut = new PrintWriter(config.getLoggingUtil().getLogDir().getAbsolutePath() + "/train-ids-fold-" + fold + ".txt");
		for (String id : trainRowIds) {				
			trainIdsOut.println(id);
		}
		trainIdsOut.flush();
		trainIdsOut.close();

		PrintWriter out = new PrintWriter(config.getLoggingUtil().getLogDir().getAbsolutePath() + "/confusion-matrix-fold-" + fold + ".txt");
		out.println(enhancedConfusionMatrix);
		out.flush();
		out.close();
		
		PrintWriter pw = new PrintWriter(config.getLoggingUtil().getLogDir().getAbsolutePath() + "/confusion-matrix-fold-" + fold + ".csv");
		pw.println(enhancedConfusionMatrix.toCsv(","));
		pw.flush();
		pw.close();
		
		int requestedWords = config.getNrTopWords(LDAConfiguration.NO_TOP_WORDS_DEFAULT);
		if(!fakeTextData && requestedWords>dolda.getAlphabet().size()) {
			requestedWords = dolda.getAlphabet().size();
		}

		PrintWriter topOut = new PrintWriter(lgDir.getAbsolutePath() + "/fold-" + fold + "-TopWords.txt");
		if(!fakeTextData) {
			String topWords = LDAUtils.formatTopWords(
					LDAUtils.getTopWords(requestedWords, 
							dolda.getAlphabet().size(), 
							dolda.getNoTopics(), 
							dolda.getTypeTopicMatrix(), 
							dolda.getAlphabet()));
			topOut.println(topWords);
			System.out.println("Top words are: \n" + topWords);
		} else { 
			topOut.println("No text data used");
		}
		topOut.flush();
		topOut.close();
	}

	public Trial evaluateOnce(InstanceList instances, double trainingProportion, double testProportion) throws IOException {

		if(testProportion + trainingProportion != 1.0) {
			throw new IllegalArgumentException("Training proportion (" + trainingProportion + ") Test proportion (" + testProportion + ") must sum to one but sums to: " + (trainingProportion + testProportion) );
		}

		int TRAINING = 0;
		int TESTING = 1;
		//int VALIDATION = 2;

		// Split the input list into training (90%) and testing (10%) lists.                               
		// The division takes place by creating a copy of the list,                                        
		//  randomly shuffling the copy, and then allocating                                               
		//  instances to each sub-list based on the provided proportions.                                  

		InstanceList[] instanceLists =
				instances.split(new Randoms(),
						new double[] {trainingProportion, testProportion, 0.0});

		trainXs = DOLDAConfigUtils.extractXs(instanceLists[TRAINING], fullDataset.getX(), fullDataset.getRowIds());
		trainYs = DOLDAConfigUtils.extractYs(instanceLists[TRAINING], fullDataset.getY(), fullDataset.getRowIds());

		testXs = DOLDAConfigUtils.extractXs(instanceLists[TESTING], fullDataset.getX(), fullDataset.getRowIds());
		testYs = DOLDAConfigUtils.extractYs(instanceLists[TESTING], fullDataset.getY(), fullDataset.getRowIds());
		testRowIds = DOLDAConfigUtils.extractRowIds(instanceLists[TESTING], fullDataset.getX(), fullDataset.getRowIds());


		// The third position is for the "validation" set,                                                 
		//  which is a set of instances not used directly                                                  
		//  for training, but available for determining                                                    
		//  when to stop training and for estimating optimal                                               
		//  settings of nuisance parameters.                                                               
		// Most Mallet ClassifierTrainers can not currently take advantage                                 
		//  of validation sets.                                                                            

		DOLDA dolda = train( instanceLists[TRAINING] );

		if(!fakeTextData) {
			double [][] phi = dolda.getPhiMeans();

			try {
				sampleTestset(TESTING, instanceLists, phi); 

			} catch (ConfigurationException e) {
				throw new IllegalArgumentException(e);
			}
		} else {
			sampledSupervisedTestTopics = new double[testXs.length][];
			for (int i = 0; i < sampledSupervisedTestTopics.length; i++) {
				sampledSupervisedTestTopics[i] = new double[0];
			}
		}

		return new Trial(this, instanceLists[TESTING]);
	}

	protected void sampleTestset(int TESTING, InstanceList[] instanceLists, double[][] phi) throws ConfigurationException, IOException {
		SpaliasUncollapsedParallelLDA spalias = new SpaliasUncollapsedParallelLDA(new DOLDAPlainLDAConfiguration(config));
		spalias.addInstances(instanceLists[TESTING]);
		spalias.setPhi(MatrixOps.clone(phi), getAlphabet(), getLabelAlphabet());
		spalias.sampleZGivenPhi(spaliasIterations);
		// sampledZbar will contain the total number of topics
		double [][] sampledTestZBar = spalias.getZbar();
		// Pick out the 'supervised' topics...
		sampledSupervisedTestTopics = MatrixOps.extractCols(0,config.getNoSupervisedTopics(-1), sampledTestZBar);
	}
	
	/* (non-Javadoc)
	 * @see xyz.lejon.bayes.models.dolda.DOLDAClassifier#getSampler()
	 */
	@Override
	public DOLDA getSampler() {
		return lastDolda;
	}
	
	/* (non-Javadoc)
	 * @see xyz.lejon.bayes.models.dolda.DOLDAClassifier#abort()
	 */
	@Override
	public boolean abort() {
		abort = true;
		return abort;
	}
	
	public List<String> getFixedTrainingIds() {
		return fixedTrainingIds;
	}

	@Override
	public void setFixedTrainingIds(List<String> fixedTestIds) {
		this.fixedTrainingIds = fixedTestIds;
	}

}
