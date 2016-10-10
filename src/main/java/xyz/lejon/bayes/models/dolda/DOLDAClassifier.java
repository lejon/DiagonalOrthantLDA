package xyz.lejon.bayes.models.dolda;

import java.util.List;

import cc.mallet.classify.Trial;
import cc.mallet.types.InstanceList;

public interface DOLDAClassifier {

	Trial[] crossValidate(InstanceList instances, int folds) throws Exception;

	DOLDA getSampler();
	
	void setFixedTrainingIds(List<String> trainingIds);

	boolean abort();

}