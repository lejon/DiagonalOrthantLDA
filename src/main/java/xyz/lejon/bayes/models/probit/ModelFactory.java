package xyz.lejon.bayes.models.probit;

import java.lang.reflect.InvocationTargetException;

import xyz.lejon.configuration.DOConfiguration;


public class ModelFactory {
	@SuppressWarnings("unchecked")
	public static synchronized DOSampler get(DOConfiguration config, double [][] xs, int [] ys) {
		String model_name = config.getSamplerClass(DOConfiguration.MODEL_DEFAULT);
		
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
		argumentTypes[0] = DOConfiguration.class; 
		argumentTypes[1] = double [][].class; 
		argumentTypes[2] = int [].class;
				
		try {
			return (DOSampler) modelClass.getDeclaredConstructor(argumentTypes)
					.newInstance(config,xs,ys);
		} catch (InstantiationException | IllegalAccessException
				| InvocationTargetException
				| NoSuchMethodException | SecurityException e) {
			e.printStackTrace();
			throw new IllegalArgumentException(e);
		}
	}

}
