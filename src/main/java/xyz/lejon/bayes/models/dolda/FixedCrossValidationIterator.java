package xyz.lejon.bayes.models.dolda;

import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;

import cc.mallet.types.CrossValidationIterator;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;

public class FixedCrossValidationIterator extends CrossValidationIterator {

	private static final long serialVersionUID = 1L;
	protected List<String> fixedTrainingIds;
	protected InstanceList[] fixedFold = new InstanceList[2];
	protected boolean called = false;

	public FixedCrossValidationIterator(InstanceList ilist, int _nfolds) {
		this(ilist, _nfolds, new java.util.Random (System.currentTimeMillis ()), null);
	}

	public FixedCrossValidationIterator(InstanceList ilist, int nfolds, Random r) {
		this(ilist, nfolds, r, null);
	}

	public FixedCrossValidationIterator(InstanceList instances, int folds, Random r, List<String> fixedTrainingIds) {
		super(instances, 1, new java.util.Random (System.currentTimeMillis ()));

		int TRAINING = 0;
		int TESTING = 1;

		this.fixedTrainingIds = fixedTrainingIds;
		InstanceList training = new InstanceList(instances.getPipe());
		InstanceList test = new InstanceList(instances.getPipe());
		if(fixedTrainingIds!=null) {
			for (Instance instance : instances) {
				if(fixedTrainingIds.contains(instance.getName().toString())) {
					training.add(instance);
				} else {
					test.add(instance);
				}
			}
		}
		fixedFold[TRAINING] = training;
		fixedFold[TESTING] = test;
	}

	@Override
	public InstanceList[] nextSplit() {
		if (called) {
            throw new NoSuchElementException();
        }
        
        called = true;
        
        return fixedFold;
	}

	@Override
	public InstanceList[] nextSplit(int numTrainFolds) {
		if(numTrainFolds!=1) throw new UnsupportedOperationException ();
		return nextSplit();
	}

	@Override
	public boolean hasNext() {
		return !called;
	}
}
