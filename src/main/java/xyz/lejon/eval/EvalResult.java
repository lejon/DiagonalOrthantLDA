package xyz.lejon.eval;

public class EvalResult {
	public int [] predictedLabels;
	public int noCorrect;
	public int [][] confusionMatrix;
	
	public EvalResult(int[] predictedLabels, int noCorrect) {
		super();
		this.predictedLabels = predictedLabels;
		this.noCorrect = noCorrect;
	}

	public EvalResult(int[] predictedLabels, int noCorrect, int [][] confusionMatrix) {
		super();
		this.predictedLabels = predictedLabels;
		this.noCorrect = noCorrect;
		this.confusionMatrix = confusionMatrix;
	}

	public int[] getPredictedLabels() {
		return predictedLabels;
	}

	public void setPredictedLabels(int[] predictedLabels) {
		this.predictedLabels = predictedLabels;
	}

	public int getNoCorrect() {
		return noCorrect;
	}

	public void setNoCorrect(int noCorrect) {
		this.noCorrect = noCorrect;
	}

	public int[][] getConfusionMatrix() {
		return confusionMatrix;
	}

	public void setConfusionMatrix(int[][] confusionMatrix) {
		this.confusionMatrix = confusionMatrix;
	}
}
