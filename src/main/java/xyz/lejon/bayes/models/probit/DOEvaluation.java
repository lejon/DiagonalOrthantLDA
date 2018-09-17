package xyz.lejon.bayes.models.probit;

import java.util.Collection;
import java.util.Map;

import joinery.DataFrame;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import xyz.lejon.eval.EvalResult;
import xyz.lejon.utils.MatrixOps;

public class DOEvaluation {
	
	public static EvalResult evaluate(double[][] xs, int[] ys, double[][] betas) {
		return evaluate(xs,ys,betas,false);
	}

	public static EvalResult evaluate(double[][] xs, int[] ys, double[][] betas, boolean verbose) {
		return evaluateMaxA(xs,ys,betas,verbose);
	}

	protected static EvalResult evaluateMaxA(double[][] xs, int[] ys, double[][] betas, boolean verbose) {
		int noCorrect = 0;
		int noClassesInTrainingset = betas.length;
		int [] predClass = new int[xs.length];
		int [] realClass = new int[xs.length];
		for (int row = 0; row < xs.length; row++) {
			double [][] xrow = new double[1][xs[row].length];
			xrow[0] = xs[row];
			DenseMatrix64F xrowd = new DenseMatrix64F(xrow);
			DenseMatrix64F Betas = new DenseMatrix64F(betas);
			DenseMatrix64F aHatd = new DenseMatrix64F(xrowd.numRows,Betas.numRows);
			CommonOps.multTransB(xrowd, Betas, aHatd);
			double [] aHat = MatrixOps.extractDoubleVector(aHatd);
			predClass[row] = MatrixOps.maxIdx(aHat);
			realClass[row] = ys[row];
			if(predClass[row]==realClass[row]) {
				if(verbose) System.out.println(row + ": True: " + realClass[row] + " Predicted: " + predClass[row] + " => Correct!");
				noCorrect++;
			} else {
				if(verbose) System.out.println(row + ": True: " + realClass[row] + " Predicted: " + predClass[row] + " => Incorrect!");
			}
		}
		int [][] confusionMatrix = buildConfusionMatrix(realClass, predClass, noClassesInTrainingset);
		return new EvalResult(predClass, noCorrect, confusionMatrix);
	}
	
	public static EvalResult predict(double[][] xs, double[][] betas) {
		return predictMaxA(xs,betas);
	}
	
	protected static EvalResult predictMaxA(double[][] xs, double[][] betas) {
		int [] predClass = new int[xs.length];
		for (int row = 0; row < xs.length; row++) {
			double [][] xrow = new double[1][xs[row].length];
			xrow[0] = xs[row];
			DenseMatrix64F xrowd = new DenseMatrix64F(xrow);
			DenseMatrix64F Betas = new DenseMatrix64F(betas);
			DenseMatrix64F aHatd = new DenseMatrix64F(xrowd.numRows,Betas.numRows);
			CommonOps.multTransB(xrowd, Betas, aHatd);
			double [] aHat = MatrixOps.extractDoubleVector(aHatd);
			predClass[row] = MatrixOps.maxIdx(aHat);
		}
		return new EvalResult(predClass, -1, null);
	}
	
	static int[][] buildConfusionMatrix(int[] realClass, int[] predClass, int noClassesInTrainingset) {
		if(realClass.length!=predClass.length) 
			throw new IllegalArgumentException("Predicted and real class vectors are not of equal length: " + predClass.length + " != "  + realClass.length);
		int [][] result = new int[noClassesInTrainingset][noClassesInTrainingset];
		for (int i = 0; i < predClass.length; i++) {
			int predicted = predClass[i];
			int real = realClass[i];
			result[predicted][real] += 1;
		}
		return result;
	}
	public static String confusionMatrixToString(int [][] confusionMatrix, Map<Integer, String> idMap) {
		return confusionMatrixToCSV(confusionMatrix, idMap, ", ");
	}
	
	public static String confusionMatrixToCSV(int [][] confusionMatrix, Map<Integer, String> idMap, String colSep) {
		String result = "";
		
		// Find the widest column
		int maxlen = 0;
		Collection<String> vals = idMap.values();
		for (String string : vals) {
			if(string.length()>maxlen) {
				maxlen = string.length();
			}
		}
		
		result += spaces(maxlen + colSep.length());
		int [] colWidth = new int[confusionMatrix[0].length];
		for (int col = 0; col < confusionMatrix[0].length; col++) {
			String className = idMap.get(col);
			result += className;
			if(col+1!=confusionMatrix[0].length) 
				result += colSep;
			colWidth[col] = className.length() + colSep.length();
		}
		result += "\n";

		for (int row = 0; row < confusionMatrix.length; row++) {
			String rowClassName = idMap.get(row);
			result += rowClassName + spaces(maxlen - rowClassName.length() + colSep.length());
			for (int col = 0; col < confusionMatrix[0].length; col++) {
				result += confusionMatrix[row][col] + " ";
				result += spaces(colWidth[col] - (confusionMatrix[row][col] + " ").length());
			}
			result += rowClassName + "\n";
		}
		
		return result;
	}
	
	public static DataFrame<Object> confusionMatrixToDataFrame(int [][] confusionMatrix, Map<Integer, String> idMap) {
		
		/*String [] columnLabels = new String[confusionMatrix[0].length+1];
		columnLabels[0] = "Class";
		for (int lblIdx = 1; lblIdx < columnLabels.length; lblIdx++) {
			columnLabels[lblIdx] = idMap.get(lblIdx-1);
		}
		
		DataFrame<Object> out = new DataFrame<>(columnLabels);
		List<double []> [] sampledBetas = dolda.getSampledBetas();
		for (int k = 0; k < sampledBetas.length; k++) {
			List<double []> betasClassK = sampledBetas[k];
			for (int j = 0; j < betasClassK.size(); j++) {
				double [] betaRow = betasClassK.get(j);
				List<Object> row = new ArrayList<>();
				//Add the class id to the first column
				row.add(idMap.get(k));
				for (int covariate = 0; covariate < betaRow.length; covariate++) {
					row.add(betaRow[covariate]);
				}
				out.append(row);
			}
		}
		
		
		String result = "";
		int [] tabs = new int[confusionMatrix[0].length];
		for (int col = 0; col < confusionMatrix[0].length; col++) {
			String className = idMap.get(col);
			result += className;
			if(col+1!=confusionMatrix[0].length) 
				result += ", ";
			tabs[col] = className.length();
		}
		
		result += "\n";

		for (int row = 0; row < confusionMatrix.length; row++) {
			for (int col = 0; col < confusionMatrix[0].length; col++) {
				result += confusionMatrix[row][col] + " ";
				result += spaces(tabs[col]);
			}
			result += idMap.get(row) + "\n";
		}
		*/
		return null;
	}
	
	static String spaces(int noSpace) {
		String result = "";
		for (int i = 0; i < noSpace; i++) {
			result += " ";
		}
		return result;
	}

	// Can you create short javadocs that describe what each function does? I don't understand this one.
	public static EvalResult evaluateMaxProb(double[][] xs, int[] ys, double[][] betas, boolean verbose) {
		int noCorrect = 0;
		int [] predictedLabels = new int[xs.length];
		for (int row = 0; row < xs.length; row++) {
			double [] probs = AbstractDOSampler.getClassProbabilities(xs[row], betas);
			System.out.println(ys[row] + "=> Probs: " + MatrixOps.arrToStr(probs, "Probs" ));
			int predClass = MatrixOps.maxIdx(probs);
			int realClass = ys[row];
			predictedLabels[row] = predClass;
			if(predClass==realClass) {
				if(verbose) System.out.println(row + ": True: " + realClass + " Predicted: " + predClass + " => Correct!");
				noCorrect++;
			} else {
				if(verbose) System.out.println(row + ": True: " + realClass + " Predicted: " + predClass + " => Incorrect!");
			}
		}
		return new EvalResult(predictedLabels, noCorrect);
	}


}
