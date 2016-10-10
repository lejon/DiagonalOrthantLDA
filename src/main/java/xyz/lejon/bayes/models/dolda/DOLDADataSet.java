package xyz.lejon.bayes.models.dolda;

import java.util.Map;

import joinery.DataFrame;
import xyz.lejon.utils.DataSet;
import xyz.lejon.utils.PrincipalComponentAnalysis;
import cc.mallet.types.InstanceList;


public class DOLDADataSet {
	
	DataSet additionalCovariates;
	InstanceList textData;
	
	public DOLDADataSet(DataSet additionalCovariates, InstanceList textData) {
		this.additionalCovariates = additionalCovariates;
		this.textData = textData;
	}

	public String [] getColnamesX() {
		return additionalCovariates.getColnamesX();
	}
	
	public double[][] getX() {
		return additionalCovariates.getX();
	}

	public double[] getXRow(int row) {
		return additionalCovariates.getX()[row];
	}

	public void setX(double[][] x) {
		additionalCovariates.setX(x);
	}

	public int[] getY() {
		return additionalCovariates.getY();
	}

	public int getYRow(int row) {
		return additionalCovariates.getY()[row];
	}

	public String getRowId(int row) {
		return additionalCovariates.getIds()[row];
	}

	public String [] getRowIds() {
		return additionalCovariates.getIds();
	}

	public void setY(int[] y) {
		additionalCovariates.setY(y);
	}

	public String[] getLabels() {
		return additionalCovariates.getLabels();
	}

	public String getLabelForRow(int row) {
		return additionalCovariates.getLabels()[row];
	}

	public void setLabels(String[] labels) {
		additionalCovariates.setLabels(labels);
	}

	public Map<String, Integer> getLabelToId() {
		return additionalCovariates.getLabelToId();
	}

	public void setLabelToId(Map<String, Integer> labelToId) {
		additionalCovariates.setLabelToId(labelToId);
	}

	public Map<Integer, String> getIdToLabels() {
		return additionalCovariates.getIdToLabels();
	}

	public void setIdToLabels(Map<Integer, String> idToLabels) {
		additionalCovariates.setIdToLabels(idToLabels);
	}

	public PrincipalComponentAnalysis getPca() {
		return additionalCovariates.getPca();
	}

	public DataFrame<Object> getOrigData() {
		return additionalCovariates.getOrigData();
	}

	public void setOrigData(DataFrame<Object> origData) {
		additionalCovariates.setOrigData(origData);
	}

	public int hashCode() {
		return additionalCovariates.hashCode();
	}

	public void setPca(PrincipalComponentAnalysis pca) {
		additionalCovariates.setPca(pca);
	}

	public boolean equals(Object obj) {
		return additionalCovariates.equals(obj);
	}

	public String toString() {
		return additionalCovariates.toString();
	}

	public InstanceList getTextData() {
		return textData;
	}

	public void setTextData(InstanceList textData) {
		this.textData = textData;
	}
	
	public boolean isEmpty() {
		return additionalCovariates == null && textData == null;
	}
}
