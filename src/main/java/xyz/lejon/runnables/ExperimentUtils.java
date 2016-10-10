package xyz.lejon.runnables;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import joinery.DataFrame;
import cc.mallet.util.LDAUtils;

public class ExperimentUtils {

	public static void saveDocTopicMeans(File lgDir, double [][] means, String documentTopicMeansOutputFilename) throws FileNotFoundException,
	IOException {
		String docTopicMeanFn = documentTopicMeansOutputFilename;
		if(means!=null && means.length>0 && means[0].length>0)
			LDAUtils.writeASCIIDoubleMatrix(means, lgDir.getAbsolutePath() + "/" + docTopicMeanFn, ",");
		else 
			System.err.println("WARNING: DOLDAClassification: asked to write doc topic means but it is empty");
	}

	public static void saveBetaSamples(File lgDir, String [] xColnames, int noColumns,
			List<double []> [] sampledBetas, Map<Integer, String> idMap, 
			String betaSamplesOutputFn) throws IOException {
		String[] columnLabels = extractColumnLabels(xColnames, noColumns);
		DataFrame<Object> out = new DataFrame<>(columnLabels);
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
		if(out.length()>0) out.writeCsv(lgDir.getAbsolutePath() + "/" + betaSamplesOutputFn);
	}

	public static void saveBetas(File lgDir, String [] xColnames, int noColumns, 
			double[][] betas, Map<Integer, String> idMap, String betasOutputFn) throws IOException {
		String[] columnLabels = extractColumnLabels(xColnames, noColumns);
		DataFrame<Object> out = new DataFrame<>(columnLabels);
		for (int j = 0; j < betas.length; j++) {
			List<Object> row = new ArrayList<>();
			//Add the class id to the first column
			row.add(idMap.get(j));
			for (int k = 0; k < betas[j].length; k++) {
				row.add(betas[j][k]);
			}
			out.append(row);
		}
		out.writeCsv(lgDir.getAbsolutePath() + "/" + betasOutputFn);
	}
	
	public static String[] extractColumnLabels(String [] xColnames, int noColumns) {
		String [] columnLabels = new String[xColnames.length+1];
		columnLabels[0] = "Class";
		for (int lblIdx = 1; lblIdx < columnLabels.length; lblIdx++) {
			// In the output name colums Xn for X covariates and Zn for supervised topics
			// We have <= since we have added the "Class" column
			if(lblIdx <= noColumns) {
				columnLabels[lblIdx] = xColnames[lblIdx-1];
			} else {
				columnLabels[lblIdx] = "Z" + (lblIdx-noColumns);
			}
		}
		return columnLabels;
	}


}
