package xyz.lejon.utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class PlainCrossValidationIterator {
	
	int [][] folds;
	double [][] allXs;
	double [] allYs;
	int noFolds;
	int currentTestFoldIdx = 0;
	boolean trainXcalled;
	boolean trainYcalled;
	boolean testXcalled;
	boolean testYcalled;

	public PlainCrossValidationIterator(double[][] allXs, double[] allYs, int noFolds) {
		if(noFolds<2) throw new IllegalArgumentException("Only support > 1 folds");
		this.allXs = allXs;
		this.allYs = allYs;
		this.noFolds = noFolds;
		
		List<Integer> idxs = new ArrayList<Integer>();
		for (int i = 0; i < allYs.length; i++) {
			idxs.add(i);
		}
		int foldCnt = allYs.length / noFolds;
		int leftover = allYs.length % noFolds;
		
		folds = new int[noFolds][];
		Collections.shuffle(idxs);
		for (int i = 0; i < noFolds; i++) {
			folds[i] = leftover > 0 ? new int[foldCnt+1] : new int[foldCnt];
			leftover--;
			for (int j = 0; j < folds[i].length && idxs.size()>0; j++) {
				folds[i][j] = idxs.remove(0);
			}
		}		
	}
	
	public int [] getCurrentTestFoldRows() {
		return folds[currentTestFoldIdx];
	}
	
	protected void manageFoldCnter() {
		if(trainXcalled && trainYcalled && testXcalled && testYcalled) {
			trainXcalled = trainYcalled = testXcalled = testYcalled = false;
			currentTestFoldIdx++;
			System.out.println("Incremented test fold");
		}
	}
	
	public boolean haveMoreFolds() {
		return currentTestFoldIdx<noFolds;
	}

	public double[][] nextTrainX() {
		if(currentTestFoldIdx>=noFolds) throw new IllegalStateException("No more folds");
		int trainLen = 0;
		for (int i = 0; i < folds.length; i++) {
			if(i==currentTestFoldIdx) continue;
			trainLen += folds[i].length;
		}
		
		double [][] xFold = new double[trainLen][];
		int foldIdx = 0;
		for (int i = 0; i < folds.length; i++) {
			if(i==currentTestFoldIdx) continue;
			for (int j = 0; j < folds[i].length; j++) {				
				xFold[foldIdx++] = allXs[folds[i][j]];
			}
		}
		trainXcalled = true;
		manageFoldCnter();
		return xFold;
	}

	public double[] nextTrainY() {
		if(currentTestFoldIdx>=noFolds) throw new IllegalStateException("No more folds");
		int trainLen = 0;
		for (int i = 0; i < folds.length; i++) {
			if(i==currentTestFoldIdx) continue;
			trainLen += folds[i].length;
		}
		double [] yFold = new double[trainLen];
		int foldIdx = 0;
		for (int i = 0; i < folds.length; i++) {
			if(i==currentTestFoldIdx) continue;
			for (int j = 0; j < folds[i].length; j++) {
				yFold[foldIdx++] = allYs[folds[i][j]];
			}
		}
		trainYcalled = true;
		manageFoldCnter();
		return yFold;
	}

	public double[][] nextTestX() {
		if(currentTestFoldIdx>=noFolds) throw new IllegalStateException("No more folds");
		double [][] xFold = new double[folds[currentTestFoldIdx].length][];
		for (int i = 0; i < folds[currentTestFoldIdx].length; i++) {
			xFold[i] = allXs[folds[currentTestFoldIdx][i]];
		}
		testXcalled = true;
		manageFoldCnter();
		return xFold;
	}

	public double[] nextTestY() {
		if(currentTestFoldIdx>=noFolds) throw new IllegalStateException("No more folds");
		double [] yFold = new double[folds[currentTestFoldIdx].length];
		for (int i = 0; i < folds[currentTestFoldIdx].length; i++) {
			yFold[i] = allYs[folds[currentTestFoldIdx][i]];
		}
		testYcalled = true;
		manageFoldCnter();
		return yFold;
	}

}
