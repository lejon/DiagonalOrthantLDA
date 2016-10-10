package xyz.lejon.utils;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import joinery.DataFrame;

public class RegressionDataSet {
	public double [][] X;
	public double [] Y;
	public String [] labels;
	public Map<String,Integer> labelToId = new HashMap<>();
	public Map<Integer, String> idToLabels = new HashMap<>();
	DataFrame<Object> origData;
	DataFrame<Number> mmdf;
	PrincipalComponentAnalysis pca;
	boolean addedIntercept = false;
	
	public RegressionDataSet(DataFrame<Object> df, double [][] xs, double[] ys, String[] labels, Map<String, Integer> labelToId,
			Map<Integer, String> idToLabels, PrincipalComponentAnalysis pca, boolean addedInterceopt) {
		super();
		this.pca = pca;
		origData = df;
		X = xs;
		Y = ys;
		this.labels = labels;
		this.labelToId = labelToId;
		this.idToLabels = idToLabels;
		this.addedIntercept = addedInterceopt;
	}

	public double[][] getX() {
		return X;
	}

	public void setX(double[][] x) {
		X = x;
	}

	public double[] getY() {
		return Y;
	}

	public void setY(double[] y) {
		Y = y;
	}

	public String[] getLabels() {
		return labels;
	}

	public String[] getOrigColNames() {
		Set<Object> cols = origData.columns();
		int interceptSize = addedIntercept ? 1 : 0;
		String [] res = new String[cols.size()+interceptSize];
		int cnt = 0;
		if(addedIntercept) { 
			res[cnt++] = "Intercept";
		}
		for (Object col : cols) {
			res[cnt++] = col.toString();
		}
		return res;
	}

	/**
	 * These are the column names after dummy variable expansion
	 * 
	 * @return transformed column names
	 */
	public String[] getTransformedColNames() {
		Set<Object> cols = mmdf.columns();
		int interceptSize = addedIntercept ? 1 : 0;
		String [] res = new String[cols.size()+interceptSize];
		int cnt = 0;
		if(addedIntercept) { 
			res[cnt++] = "Intercept";
		}
		for (Object col : cols) {
			res[cnt++] = col.toString();
		}
		return res;
	}

	public void setLabels(String[] labels) {
		this.labels = labels;
	}

	public Map<String, Integer> getLabelToId() {
		return labelToId;
	}

	public void setLabelToId(Map<String, Integer> labelToId) {
		this.labelToId = labelToId;
	}

	public Map<Integer, String> getIdToLabels() {
		return idToLabels;
	}

	public void setIdToLabels(Map<Integer, String> idToLabels) {
		this.idToLabels = idToLabels;
	}

	public PrincipalComponentAnalysis getPca() {
		return pca;
	}
	
	public DataFrame<Object> getOrigData() {
		return origData;
	}

	public void setOrigData(DataFrame<Object> origData) {
		this.origData = origData;
	}
	
	public DataFrame<Number> getTransformedData() {
		return mmdf;
	}

	public void setTransformedData(DataFrame<Number> transformedData) {
		this.mmdf = transformedData;
	}

	public void setPca(PrincipalComponentAnalysis pca) {
		this.pca = pca;
	}

	public void setAddedIntercept(boolean b) {
		addedIntercept = b;
	}
	
	public boolean addedIntercept() {
		return addedIntercept;
	}
}
