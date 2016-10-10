package xyz.lejon.configuration;

import java.util.ArrayList;
import java.util.List;

import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;

public class DOLDAConfigUtils {
	
	public static List<String> extractLDALabels(InstanceList testInstances) {
		// Gotta be a better way to do this, but I cannot figure out how...
		List<String> ldaLabels = new ArrayList<String>();
		for (Instance instance : testInstances) {
			ldaLabels.add(instance.getTarget().toString());
		}
		return ldaLabels;
	}

	public static List<String> extractLDAIds(InstanceList testInstances) {
		List<String> ldaIds = new ArrayList<String>();
		for (Instance instance : testInstances) {
			String instanceName = instance.getName().toString();
			if(ldaIds.contains(instanceName)) throw new IllegalArgumentException("DOLDAConfigUtils: Observation ID's are not unique! " + instanceName + " is already in id set...");
			ldaIds.add(instanceName);
		}
		return ldaIds;
	}
	
	public static double[][] extractXs(InstanceList trainingSet, double[][] x, String[] rowIds) {
		double [][] result = new double[trainingSet.size()][];
		int copied = 0;
		for(Instance instance : trainingSet) {
			String instanceId = instance.getName().toString();
			for (int i = 0; i < rowIds.length; i++) {
				if(rowIds[i].equals(instanceId)) {
					result[copied++] = x[i].clone();
					// Found it, no use looping through the rest!
					break;
				}
			}
		}
		return result;
	}
	
	public static String [] extractRowIds(InstanceList trainingSet, double[][] x, String[] rowIds) {
		String [] result = new String[trainingSet.size()];
		int copied = 0;
		for(Instance instance : trainingSet) {
			result[copied++] = instance.getName().toString();
		}
		return result;
	}

	public static int [] extractYs(InstanceList trainingSet, int [] y, String[] rowIds) {
		int [] result = new int[trainingSet.size()];
		int copied = 0;
		for(Instance instance : trainingSet) {
			String instanceId = instance.getName().toString();
			for (int i = 0; i < rowIds.length; i++) {
				if(rowIds[i].equals(instanceId)) {
					result[copied++] = y[i];
					// Found it, no use looping through the rest!
					break;
				}
			}
		}
		return result;
	}

}
