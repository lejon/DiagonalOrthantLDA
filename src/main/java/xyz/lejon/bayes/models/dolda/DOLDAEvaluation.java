package xyz.lejon.bayes.models.dolda;

import java.util.Map;

import xyz.lejon.bayes.models.probit.DOEvaluation;
import xyz.lejon.eval.EvalResult;

public class DOLDAEvaluation {

	/**
	 * If text data is used, the xs must be the X part concatenated with the Z-bars
	 * @param xs
	 * @param ys
	 * @param betas
	 * @return
	 */
	public EvalResult evaluate(double [][] xs, int [] ys, double [][] betas) {
		if(betas[0].length!=xs[0].length) {
			throw new IllegalArgumentException("Xs and betas does not have compatible dimensions. X=" + xs[0].length + " betas=" + betas[0].length);
		}
		return DOEvaluation.evaluate(xs, ys, betas);
	}

	public static String confusionMatrixToString(int[][] confusionMatrix, Map<Integer,String> idMap) {
		return DOEvaluation.confusionMatrixToString(confusionMatrix, idMap);
	}

	public static String confusionMatrixToCSV(int[][] confusionMatrix, Map<Integer, String> idMap, String sep) {
		return DOEvaluation.confusionMatrixToCSV(confusionMatrix, idMap, sep);
	}

}
