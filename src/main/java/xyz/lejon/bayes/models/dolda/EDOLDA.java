package xyz.lejon.bayes.models.dolda;

import static java.lang.Math.exp;

import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.configuration.ConfigurationException;

import xyz.lejon.configuration.DOLDAConfiguration;
import xyz.lejon.utils.MatrixOps;
import cc.mallet.topics.LDADocSamplingContext;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.LabelSequence;

/**
 * This is a more efficient version of DOLDA which uses another mathematical 
 * decomposition which enables precalculation of values and then updating 
 * these on the fly during sampling. 
 * 
 * @author Leif Jonsson
 *
 */
public abstract class EDOLDA extends DOLDAGibbsSampler {

	private static final long serialVersionUID = 1L;
	
	double [][] betaSums;

	public EDOLDA(DOLDAConfiguration parentCfg, double[][] xs, int[] ys) throws ConfigurationException {
		super(parentCfg, xs, ys);
		betaSums = new double[numTopics][numTopics];
	}

	@Override
	public void preIteration() {
		super.preIteration();
		
		// Precalculate topic weights, unsupervised topics should stay 0
		// supervised topics should be calculated
		for (int k = 0; k < numTopics; k++) {
			for (int i = 0; i < numTopics; i++) {
				double weight = 0.0;
				if(isSupervised(k) && isSupervised(i)) {
					for (int l = 0; l < betas.length; l++) {
						// The beta coeff. for class l and supervised topics k and i, 
						// there are p additional covariates before the z_bar comes
						double beta_lk = betas[l][p+k]*betas[l][p+i];
						weight += beta_lk;
					}
				}
				betaSums[k][i] = weight;
			}
		}
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
		double [] zbar_d = supervisedDocTopicMeans[docId];
		
		// Additional covariates of doc
		double [] xd = xs[docId];

		double [] log_g_d_not_i = new double[ks];

		// This is done because we need to calculate g_d_not_i for pos 0, the rest will be updated later in the loop
		if(isSupervised(oneDocTopics[0])) {
			zbar_d[oneDocTopics[0]] = zbar_d[oneDocTopics[0]]-1/Nd;
		}

		// The supervised topics are the topics 0:(ks-1)
		for (int topic = 0; topic < ks; topic++) {
			log_g_d_not_i[topic] = calculateSupervisedLogScaling(docId, Nd, zbar_d, xd, topic, betas, Ast);
			if(Double.isNaN(log_g_d_not_i[topic]) || Double.isInfinite(log_g_d_not_i[topic])) { 
				throw new IllegalStateException("Got a broken g_d_not_i[topic=" + topic +"]: = " + log_g_d_not_i[topic]
						+ " docId=" + docId
						+ " nd=" + Nd
						+ " topic=" + topic
						+ " zbar_d=" + MatrixOps.arrToStr(zbar_d)
						+ " betas=" + MatrixOps.doubleArrayToPrintString(betas,50,50,10)
						+ " Ast=" + MatrixOps.doubleArrayToPrintString(Ast,5,5,10)
						);
			}

		}
		// This is done because we need to calculate g_d_not_i for pos 0, the rest will be updated later in the loop
		if(isSupervised(oneDocTopics[0])) {
			zbar_d[oneDocTopics[0]] = zbar_d[oneDocTopics[0]]+1/Nd;
		}	
		
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
				zbar_d[oldTopic] = zbar_d[oldTopic]-1/Nd;
			}

			for (int topic = 0; topic < numTopics; topic++) {
				double scaling = 1.0;
				if(isSupervised(topic)) {
					if(position>0) {						
						// In the below expression we shall use 'position-1' even if that position is an 'unsupervised topic'
						// since this is compensated for in the calculation of betaSums
						log_g_d_not_i[topic] = log_g_d_not_i[topic] + ((betaSums[topic][oldTopic]-betaSums[topic][oneDocTopics[position-1]]) / (Nd*Nd)); 
					}
					scaling = log_g_d_not_i[topic];
					scaling = exp(scaling);
					// If scaling goes "overboard" set it to 'a large value'
					if(Double.isInfinite(scaling)) scaling = 10_000;
				}
				score = (localTopicCounts[topic] + alpha) * phi[topic][type] * scaling;
				if(score<0.0 || Double.isNaN(score)) { 
					if(score<0.0 || Double.isNaN(score)) { 
						throw new IllegalStateException("Got a broken score: " 
								+ " docId=" + docId
								+ " score=" + score
								+ " Nd=" + Nd
								+ " topic=" + topic
								+ " phi[topic][type]=" + phi[topic][type]
								+ " localTopicCounts=" + MatrixOps.arrToStr(localTopicCounts) + "\n"
								+ " betas=" + MatrixOps.doubleArrayToPrintString(betas,50,50,10)
								+ " A's=" + MatrixOps.doubleArrayToPrintString(Ast,5,5,10)
								);
					}
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
				throw new IllegalStateException("UncollapsedParallelLDA: "
						+ " New sampled topic (" + newTopic + ") is not valid, valid topics are between 0 and "	+ (numTopics-1) + "." 
						+ " Score is: " + sum
						+ " docId=" + docId
						+ " localTopicCounts=" + MatrixOps.arrToStr(localTopicCounts) + "\n"
						+ " docLength=" + docLength
						);
			}

			// Put that new topic into the counts
			oneDocTopics[position] = newTopic;
			localTopicCounts[newTopic]++;
			// Add its contribution to z_bar
			if(isSupervised(newTopic)) {
				zbar_d[newTopic] = zbar_d[newTopic]+1/Nd;
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
}
