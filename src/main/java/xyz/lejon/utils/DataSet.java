package xyz.lejon.utils;

import java.util.HashMap;
import java.util.Map;

import joinery.DataFrame;

public class DataSet {
	public double [][] X;
	public int [] Y;
	public String [] labels;
	public String [] ids;

	public Map<String,Integer> labelToId = new HashMap<>();
	public Map<Integer, String> idToLabels = new HashMap<>();
	DataFrame<Object> origData;
	String [] colnamesX;
	PrincipalComponentAnalysis pca;
	
	public DataSet(DataFrame<Object> df, double [][] xs, int[] ys, String[] labels, Map<String, Integer> labelToId,
			Map<Integer, String> idToLabels, PrincipalComponentAnalysis pca, String [] ids) {
		super();
		this.pca = pca;
		origData = df;
		X = xs;
		Y = ys;
		this.labels = labels;
		this.labelToId = labelToId;
		this.idToLabels = idToLabels;
		this.ids = ids;
	}
	
	public String [] getColnamesX() {
		return colnamesX;
	}

	public void setColnamesX(String [] colnamesX) {
		this.colnamesX = colnamesX;
	}

	public double[][] getX() {
		return X;
	}

	public String [] getIds() {
		return ids;
	}
	
	public void setIds(String[] ids) {
		this.ids = ids;
	}

	public void setX(double[][] x) {
		X = x;
	}

	public int[] getY() {
		return Y;
	}

	public void setY(int[] y) {
		Y = y;
	}

	public String[] getLabels() {
		return labels;
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

	public void setPca(PrincipalComponentAnalysis pca) {
		this.pca = pca;
	}
}
