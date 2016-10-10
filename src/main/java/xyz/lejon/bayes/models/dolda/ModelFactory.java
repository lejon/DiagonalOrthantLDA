package xyz.lejon.bayes.models.dolda;

import java.lang.reflect.InvocationTargetException;

import xyz.lejon.configuration.DOLDAConfiguration;


public class ModelFactory {
	@SuppressWarnings("unchecked")
	public static synchronized DOLDA get(DOLDAConfiguration config, double[][] xs, int[] ys) {
		String model_name = config.getSamplerClass(DOLDAConfiguration.MODEL_DEFAULT);

		@SuppressWarnings("rawtypes")
		Class modelClass = null;
		try {
			modelClass = Class.forName(model_name);
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
			throw new IllegalArgumentException(e);
		}

		@SuppressWarnings("rawtypes")
		Class[] argumentTypes = new Class[3];
		argumentTypes[0] = DOLDAConfiguration.class; 
		argumentTypes[1] = double [][].class; 
		argumentTypes[2] = int [].class;

		try {
			return (DOLDA) modelClass.getDeclaredConstructor(argumentTypes)
					.newInstance(config,xs,ys);
		} catch (InstantiationException | IllegalAccessException
				| InvocationTargetException
				| NoSuchMethodException | SecurityException e) {
			e.printStackTrace();
			throw new IllegalArgumentException(e);
		}
	}

	@SuppressWarnings("unchecked")
	public static synchronized DOLDAWithCallback get(DOLDAConfiguration config, double[][] xs, 
			int[] ys, DOLDAIterationCallback callback) {
		String model_name = config.getSamplerClass(DOLDAConfiguration.MODEL_DEFAULT);

		@SuppressWarnings("rawtypes")
		Class modelClass = null;
		try {
			modelClass = Class.forName(model_name);
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
			throw new IllegalArgumentException(e);
		}

		@SuppressWarnings("rawtypes")
		Class[] argumentTypes = new Class[3];
		argumentTypes[0] = DOLDAConfiguration.class; 
		argumentTypes[1] = double [][].class; 
		argumentTypes[2] = int [].class;

		try {
			DOLDAWithCallback dwc = (DOLDAWithCallback) modelClass.getDeclaredConstructor(argumentTypes)
					.newInstance(config,xs,ys);
			dwc.setCallback(callback);
			return dwc;
		} catch (InstantiationException | IllegalAccessException
				| InvocationTargetException
				| NoSuchMethodException | SecurityException e) {
			e.printStackTrace();
			throw new IllegalArgumentException(e);
		}
	}
}
