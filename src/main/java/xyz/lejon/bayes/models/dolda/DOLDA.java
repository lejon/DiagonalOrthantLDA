package xyz.lejon.bayes.models.dolda;
import java.io.IOException;
import java.util.List;

import cc.mallet.topics.AbortableSampler;
import cc.mallet.types.Alphabet;
import cc.mallet.types.InstanceList;

public interface DOLDA extends AbortableSampler {
	void setRandomSeed(int commonSeed);

	void addInstances(InstanceList instances);

	int getStartSeed();

	void sample (int iterations) throws IOException;

	int getNoTopics();

	double[][] getBetas();

	List<double[]>[] getSampledBetas();

	double[][] getSupervsedTopicIndicatorMeans();

	int getCurrentIteration();

	double[][] getPhi();
	
	double[][] getPhiMeans();

	double[][] getZbar();

	Alphabet getAlphabet();

	int[][] getTypeTopicMatrix();
}
