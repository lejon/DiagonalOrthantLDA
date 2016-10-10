package xyz.lejon.bayes.models.regression;

import java.lang.reflect.InvocationTargetException;

import xyz.lejon.configuration.OLSConfiguration;


public class ModelFactory {
	@SuppressWarnings("unchecked")
	public static synchronized LinearRegression get(OLSConfiguration config, double [][] xs, double [] ys) {
		String model_name = config.getSamplerClass(OLSConfiguration.MODEL_DEFAULT);
		
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
		argumentTypes[0] = OLSConfiguration.class; 
		argumentTypes[1] = double [][].class; 
		argumentTypes[2] = double [].class;
				
		try {
			return (LinearRegression) modelClass.getDeclaredConstructor(argumentTypes)
					.newInstance(config,xs,ys);
		} catch (InstantiationException | IllegalAccessException
				| InvocationTargetException
				| NoSuchMethodException | SecurityException e) {
			e.printStackTrace();
			throw new IllegalArgumentException(e);
		}
	}

}
