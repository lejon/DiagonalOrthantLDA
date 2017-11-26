package xyz.lejon.bayes.models.dolda;

import static java.lang.Math.exp;
import static java.lang.Math.pow;
import static xyz.lejon.utils.MatrixOps.dot2P1;

import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.configuration.ConfigurationException;
import org.jblas.DoubleMatrix;

import cc.mallet.topics.LDADocSamplingContext;
import cc.mallet.topics.LogState;
import cc.mallet.topics.UncollapsedParallelLDA;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.IDSorter;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelSequence;
import cc.mallet.util.LDAUtils;
import xyz.lejon.bayes.models.probit.AbstractDOSampler;
import xyz.lejon.configuration.DOLDAConfiguration;
import xyz.lejon.configuration.DOLDAPlainLDAConfiguration;
import xyz.lejon.sampling.NegativeTruncatedNormal;
import xyz.lejon.sampling.PositiveTruncatedNormal;
import xyz.lejon.utils.MatrixOps;

/**
 * This is a Supervised LDA version with optional extra covariates. The sampler learns
 * the topics and the extra covariates at the same time so both topics and betas for the
 * additional covariates are used to guide the training.
 * 
 * The supervised part uses the computationally very efficient Diagonal Orthant probit model. 
 * 
 * @author Leif Jonsson
 *
 */
public abstract class DOLDAGibbsSampler extends UncollapsedParallelLDA implements DOLDA, DOLDAWithCallback {

	DOLDAIterationCallback iterationCallback = null;
	private static final long serialVersionUID = 1L;

	double Sigma = 1;
	int noClasses = -1;
	int ks = 0; // No supervised Topics
	int p = 0;  // No other covariates 
	double [][] betas; // Indexed by [class][covariates + zbar]
	double [][]xs;  
	int [] ys;
	Object betaLock = new Object();
	boolean useIntecept = true;
	
	boolean saveBetaSamples = false;
	boolean useUnsupervisedLL = false;

	protected PositiveTruncatedNormal ptn = new PositiveTruncatedNormal();
	protected NegativeTruncatedNormal ntn = new NegativeTruncatedNormal();

	// Utilities transposed 
	protected double [][] Ast; // Indexed by [class][document]
	// document topic means (z_bar)
	protected double[][] supervisedDocTopicMeans; // Indexed by [docId][topic]

	private ForkJoinPool averagerPool;

	// Array of lists of sampled betas. The array is noClasses long, each list contains the sampled betas for that class.
	List<double []> [] sampledBetas;

	protected int betaSamplesToSave = 200;

	@SuppressWarnings("unchecked")
	public DOLDAGibbsSampler(DOLDAConfiguration parentCfg, double [][]xs,  int [] ys) throws ConfigurationException {
		super(new DOLDAPlainLDAConfiguration(parentCfg));
		
		useIntecept = parentCfg.getUseIntercept();
		
		saveBetaSamples = parentCfg.saveBetaSamples();
		
		noClasses = parentCfg.getLabelMap().keySet().size();
		
		if(parentCfg.likelihoodType()!=null) {
			useUnsupervisedLL = parentCfg.likelihoodType().toLowerCase().startsWith("unsupervised");
		}

		averagerPool = new ForkJoinPool();

		p = xs[0].length;

		Ast  = new double[noClasses][xs.length];
		
		// Important to initialize A's to reasonable values
		for (int k = 0; k < noClasses; k++) {
			for (int row = 0; row < ys.length; row++) {
				double mean = 0;
				// Sample Z_i,y_i | B_y_i ~ N_+( x_i' * Beta, 1)
				if(k==ys[row]) {
					// Positive Truncated Normal
					Ast[k][row] = ptn.rand(mean, 1);
				} else {
					// Negative Truncated Normal
					Ast[k][row] = ntn.rand(mean, 1);
				}
			}
		}

		if(parentCfg.getTextDatasetTrainFilename()!=null && parentCfg.getTextDatasetTrainFilename().trim().length()>0) {
			ks = parentCfg.getNoSupervisedTopics(ks);
		} else {
			ks = 0;
		}
		if(ks>numTopics) 
			throw new IllegalArgumentException("Cannot have more supevised topics (" + ks + "), than total number of topics (" + numTopics + ")");
		if(ks<0) 
			throw new IllegalArgumentException("Cannot have negative number of supevised topics (" + ks + ")");
		
		System.out.println("Using " + ks + " supervised topics");
		supervisedDocTopicMeans = new double[ys.length][ks];

		this.xs = xs;
		this.ys = ys;
		
		int xCovs = useIntecept ? xs[0].length-1 : xs[0].length; 
		if(xCovs+ks<2)
			throw new IllegalArgumentException("Have no supervised topics (" + ks + ") and no additional covariates! Did you forget to set 'supervised_topics' in the config?");

		System.out.println("Have " + noClasses + " classes");
		//betas = MatrixOps.rnorm(noClasses, p+ks);
		betas = new double[noClasses][p+ks];
		sampledBetas = new LinkedList[noClasses];
		for (int i = 0; i < sampledBetas.length; i++) {
			sampledBetas[i] = new LinkedList<double []>();
		}
	}

	@Override
	public void addInstances(InstanceList instances) {
		if(instances.getAlphabet().size()>0) {
			super.addInstances(instances);
			computeDocTopicAverages();
		}
	}

	public double [][] getSupervsedTopicIndicatorMeans() {
		return supervisedDocTopicMeans;
	}

	abstract protected void sampleBetas(double [][] Xs, double [][] XtXs, double [][] Xts, double [][] precisions);

	protected void saveBetaSample(int k) {
		if(saveBetaSamples) {
			sampledBetas[k].add(betas[k]);
			if(sampledBetas[k].size()>betaSamplesToSave) {
				sampledBetas[k].remove(0);
			}
		}
	}
	
	@Override
	public void sample(int iterations) throws IOException {
		// Only do Z sampling if we have text data
		// If we do Z sampling, the probit sampling is done in the preIteration method
		if(data.size()>0) {
			super.sample(iterations);
		} else {
			for (int iteration = 1; iteration <= iterations && !abort; iteration++) {
				probitSampling();
				if (showTopicsInterval > 0 && iteration % showTopicsInterval == 0) {
					System.out.println("Iteration " + iteration );		
					double logLik = modelLogLikelihood();	
					String loggingPath = config.getLoggingUtil().getLogDir().getAbsolutePath();
					LogState logState = new LogState(logLik, iteration, null, loggingPath, logger);
					LDAUtils.logLikelihoodToFile(logState);
					logger.info("<" + iteration + "> Log Likelihood: " + logLik);					
				}
				if(iterationCallback!=null) {
					iterationCallback.iterationState(new SimpleDOLDAIterationState(null, betas));
				}
			}
		}
	}

	@Override
	public void preIteration() {
		super.preIteration();
		probitSampling();
	}

	protected void probitSampling() {
		// Concatenate together additional covariates and mean topic indicators
		double [][] zbar_d = getSupervsedTopicIndicatorMeans();
		double [][] Xs = MatrixOps.concatenate(xs,zbar_d);
		double [][] Xts = MatrixOps.transposeSerial(Xs);
		DoubleMatrix X = new DoubleMatrix(Xs);
		DoubleMatrix XtX = X.transpose().mmul(X);
	
		//long beforeBetaSampling = System.currentTimeMillis();
		sampleBetas(Xs, XtX.toArray2(), Xts, XtX.toArray2());
		if(saveBetaSamples) {
			for (int k = 0; k < noClasses; k++) {
				saveBetaSample(k);
			}
		}
		
		//long elapsedMillis = System.currentTimeMillis() - beforeBetaSampling;
		//System.out.println("Sample Betas took: " + elapsedMillis + " ms");
		//System.out.println("Betas: " + MatrixOps.doubleArrayToPrintString(betas, 5, 5, 20));

		//long beforeASampling = System.currentTimeMillis();
		for (int row = 0; row < ys.length; row++) {			
			sampleAs(row);
		}
		//elapsedMillis = System.currentTimeMillis() - beforeASampling;
		//System.out.println("Sample As took: " + elapsedMillis + " ms");		

		//System.out.println("As: " + MatrixOps.doubleArrayToPrintString(Ast, 5, 5, 20));
	}

	@Override
	public void postIteration() {
		super.postIteration();
		if(iterationCallback!=null) {
			iterationCallback.iterationState(new SimpleDOLDAIterationState(phi, betas));
		}
	}

	public void sampleAs(int row) {
		for (int k = 0; k < noClasses; k++) {
			//double mean = dot(concatenate(xs[row],docTopicMeans[row]),betas[k]);
			double mean = dot2P1(xs[row],supervisedDocTopicMeans[row],betas[k]);
			// Sample Z_i,y_i | B_y_i ~ N_+( x_i' * Beta, 1)
			if(k==ys[row]) {
				// Positive Truncated Normal
				Ast[k][row] = ptn.rand(mean, 1);
			} else {
				// Negative Truncated Normal
				Ast[k][row] = ntn.rand(mean, 1);
			}
		}
	}

	private void averageDocTopics(FeatureSequence tokens, LabelSequence topics, int docIdx) {
		final int docLength = tokens.getLength();
		if(docLength==0) return;

		int [] oneDocTopics = topics.getFeatures();

		double[] localTopicCounts = new double[numTopics];

		for (int position = 0; position < docLength; position++) {
			int topicInd = oneDocTopics[position];
			localTopicCounts[topicInd]++;
		}

		for (int k = 0; k < ks; k++) {
			supervisedDocTopicMeans[docIdx][k] = localTopicCounts[k] / docLength;
			if(Double.isInfinite(supervisedDocTopicMeans[docIdx][k]) || Double.isNaN(supervisedDocTopicMeans[docIdx][k]) || supervisedDocTopicMeans[docIdx][k] < 0) { 
				throw new IllegalStateException("docTopicMeans is broken!");  
			}
		}
	}

	class DocTopicAverager extends RecursiveAction {
		final static long serialVersionUID = 1L;
		int startDoc = -1;
		int endDoc = -1;
		int limit = 100;

		public DocTopicAverager(int startDoc, int endDoc, int ll) {
			this.limit = ll;
			this.startDoc = startDoc;
			this.endDoc = endDoc;
		}

		@Override
		protected void compute() {
			try {
				if ( (endDoc-startDoc) <= limit ) {
					for (int docIdx = startDoc; docIdx < endDoc; docIdx++) {
						FeatureSequence tokenSequence =
								(FeatureSequence) data.get(docIdx).instance.getData();
						LabelSequence topicSequence =
								(LabelSequence) data.get(docIdx).topicSequence;
						averageDocTopics(tokenSequence, topicSequence, docIdx);
					}
				}
				else {
					int range = (endDoc-startDoc);
					int startDoc1 = startDoc;
					int endDoc1 = startDoc + (range / 2);
					int startDoc2 = endDoc1;
					int endDoc2 = endDoc;
					invokeAll(new DocTopicAverager(startDoc1,endDoc1,limit),
							new DocTopicAverager(startDoc2,endDoc2,limit));
				}
			}
			catch ( Exception e ) {
				e.printStackTrace();
			}
		}
	}

	protected void computeDocTopicAverages() {
		DocTopicAverager dslr = new DocTopicAverager(0,data.size(),200);                
		averagerPool.invoke(dslr);
	}
	
	@Override
	public String topWords (int numWords) {

		StringBuilder output = new StringBuilder();

		IDSorter[] sortedWords = new IDSorter[numTypes];

		for (int topic = 0; topic < numTopics; topic++) {

			int [] typeMap = topicTypeCountMapping[topic];
			for (int token = 0; token < numTypes; token++) {
				Integer thisCount = typeMap[token];
				sortedWords[token] = new IDSorter(token, (thisCount != null) ? thisCount : 0);
			}

			Arrays.sort(sortedWords);

			output.append(topic + (isSupervised(topic) ? "S" : "") + "\t" + tokensPerTopic[topic] + "\t");
			for (int i=0; i < numWords; i++) {
				output.append(alphabet.lookupObject(sortedWords[i].getID()) + " ");
			}
			output.append("\n");
		}

		return output.toString();
	}

	@Override
	protected void sampleTopicAssignmentsParallel(LDADocSamplingContext ctx) {
		FeatureSequence tokens = ctx.getTokens();
		LabelSequence topics = ctx.getTopics();
		int myBatch = ctx.getMyBatch();
		int docId = ctx.getDocId();

		int type, oldTopic, newTopic;

		final int docLength = tokens.getLength();
		double Nd = (double) docLength;

		if(docLength==0) return;

		int [] tokenSequence = tokens.getFeatures();
		int [] oneDocTopics = topics.getFeatures();

		int[] localTopicCounts = new int[numTopics];

		// Find the non-zero words and topic counts that we have in this document
		for (int position = 0; position < docLength; position++) {
			int topicInd = oneDocTopics[position];
			localTopicCounts[topicInd]++;
		}

		double score, sum;
		double[] topicTermScores = new double[numTopics];

		// Initialize Array of z_bar's (topic means) 
		double [] zbar_not_i = supervisedDocTopicMeans[docId];

		// Additional covariates of doc
		double [] xd = xs[docId];

		//	Iterate over the words in the document
		for (int position = 0; position < docLength; position++) {
			type = tokenSequence[position];
			oldTopic = oneDocTopics[position];
			localTopicCounts[oldTopic]--;
			if(localTopicCounts[oldTopic]<0) 
				throw new IllegalStateException("Counts cannot be negative! Count for topic:" 
						+ oldTopic + " is: " + localTopicCounts[oldTopic]);

			// Propagates the update to the topic-token assignments
			/**
			 * Used to subtract and add 1 to the local structure containing the number of times
			 * each token is assigned to a certain topic. Called before and after taking a sample
			 * topic assignment z
			 */
			decrement(myBatch,oldTopic,type);
			// Now calculate and add up the scores for each topic for this word
			sum = 0.0;

			// Array of z_bar's (topic means) with topic 'topic's last contribution  removed.
			if(isSupervised(oldTopic)) {
				zbar_not_i[oldTopic] = zbar_not_i[oldTopic]-1/Nd;
			}

			for (int topic = 0; topic < numTopics; topic++) {
				double scaling = 1.0;
				if(isSupervised(topic)) {
					scaling = calculateSupervisedLogScaling(docId, Nd, zbar_not_i, xd, topic, betas, Ast);
					scaling = exp(scaling);
					// If scaling goes "overboard" set it to 'a large value'
					if(Double.isInfinite(scaling)) scaling = 10_000;
				}

				score = (localTopicCounts[topic] + alpha) * phi[topic][type] * scaling;
				if(score<0.0 || Double.isNaN(score)) { 
					throw new IllegalStateException("Got a broken score: " 
							+ " score=" + score
							+ " Nd=" + Nd
							+ " topic=" + topic
							+ " phi[topic][type]=" + phi[topic][type]
							+ " localTopicCounts=" + MatrixOps.arrToStr(localTopicCounts) + "\n"
							+ " betas=" + MatrixOps.doubleArrayToPrintString(betas,50,50,10)
							+ " A's=" + MatrixOps.doubleArrayToPrintString(Ast,5,5,10)
							);
				}
				topicTermScores[topic] = score;
				sum += score;
			}
			// Choose a random point between 0 and the sum of all topic scores
			// The thread local random performs better in concurrent situations 
			// than the standard random which is thread safe and incurs lock 
			// contention
			double U = ThreadLocalRandom.current().nextDouble();
			double sample = U * sum;

			newTopic = -1;
			while (sample > 0.0) {
				newTopic++;
				sample -= topicTermScores[newTopic];
			} 

			// Make sure we actually sampled a valid topic
			if (newTopic < 0 || newTopic >= numTopics) {
				throw new IllegalStateException("UncollapsedParallelLDA: New sampled topic ( " 
						+ newTopic + ") is not valid, valid topics are between 0 and " 
						+ (numTopics-1) + ". Score is: " + sum);
			}

			// Put that new topic into the counts
			oneDocTopics[position] = newTopic;
			localTopicCounts[newTopic]++;
			// Add its contribution to z_bar
			if(isSupervised(newTopic)) {
				zbar_not_i[newTopic] = zbar_not_i[newTopic]+1/Nd;
			}

			// Propagates the update to the topic-token assignments
			/**
			 * Used to subtract and add 1 to the local structure containing the number of times
			 * each token is assigned to a certain topic. Called before and after taking a sample
			 * topic assignment z
			 */
			increment(myBatch,newTopic,type);
		}
	}
	
	protected static double calculateSupervisedLogScaling(int docId, double nd, double[] zbar_not_is, 
			double[] xds, int topic, double[][] betas,	double[][] ast) {
		double supervision = 0.0;
		int p = xds.length;
		for (int l = 0; l < betas.length; l++) {
			// The beta coeff. for class l and topic 'topic', there are p additional covariates before the z_bar comes
			double beta_lk = betas[l][p+topic];

			// The calculated utility for class 'l' and document d
			double a_ld = ast[l][docId];
			// Estimated utility of doc

			double est = dot2P1(xds,zbar_not_is,betas[l]);
			double contrib = -2.0 * (beta_lk / nd) * (a_ld-est) + pow(beta_lk/nd,2);
			supervision += contrib;
		}

		return -supervision/2;
	}

	protected boolean isSupervised(int topic) {
		return topic < ks;
	}

	@Override
	public double[][] getBetas() {
		return betas;
	}

	@Override
	public double[][] getPhi() {
		return phi;
	}

	@Override
	public List<double []> [] getSampledBetas() {
		return sampledBetas;
	}
	
	@Override
	public void setCallback(DOLDAIterationCallback callback) {
		iterationCallback = callback;
	}
	
	/**
	 * The model loglikelihood for the DO Probit model is the loglikelihood
	 * for the original LDA model plus the contribution from the DO part.
	 * 
	 * @see cc.mallet.topics.UncollapsedParallelLDA#modelLogLikelihood()
	 */
	@Override
	public double modelLogLikelihood() {
		// Calculate the log likelihood for the usual LDA model
		double loglik = super.modelLogLikelihood();
		// Calculating the true supervised Log Likelihood is VERY time consuming
		// if we can settle with the unsupervised LL it is MUCH faster
		if(useUnsupervisedLL) return loglik;

		double DOpart_loglik = doProbitLikelihood();	
		
		loglik = loglik + DOpart_loglik;
		return loglik;
	}

	protected double doProbitLikelihood() {
		// Calculate the mean matrix (same as used in sampling beta), X%*%eta
		// Size should be D \times L (no docs times no classes). [i] is row, [j] is col
		double [][] zbar_d = getSupervsedTopicIndicatorMeans();
		return AbstractDOSampler.doProbitLogLikelihood(MatrixOps.concatenate(xs,zbar_d), ys, betas);
	}
}