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
	
}
