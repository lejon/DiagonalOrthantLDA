package xyz.lejon.configuration;

import org.apache.commons.configuration.HierarchicalINIConfiguration;

public class ConfigurationUtils {
	
	public static double [] getDoubleArrayProperty(HierarchicalINIConfiguration conf, String key) {
		String [] ints = conf.getStringArray(key);
		if(ints==null || ints.length==0) { 
			throw new IllegalArgumentException("Could not find any double array for key:" + key); 
		}
		double [] result = new double[ints.length];
		for (int i = 0; i < ints.length; i++) {
			result[i] = Double.parseDouble(ints[i].trim());
		}
		return result;
	}


}
