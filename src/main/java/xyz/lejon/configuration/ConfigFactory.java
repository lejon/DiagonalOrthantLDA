package xyz.lejon.configuration;

import org.apache.commons.configuration.ConfigurationException;

public class ConfigFactory {
	protected static Configuration mainConfig = null;
	public static Configuration getMainConfiguration(DOCommandLineParser cp) throws ConfigurationException {
		if( mainConfig == null ) {
			mainConfig = new ParsedDOConfiguration(cp);
		}
		
		return mainConfig;
	}
	
	public static Configuration getMainConfiguration(DOLDACommandLineParser cp) throws ConfigurationException {
		if( mainConfig == null ) {
			mainConfig = new ParsedDOLDAConfiguration(cp);
		}
		
		return mainConfig;
	}

	public static void resetFactory() {
		if( mainConfig != null ) {
			mainConfig = null; 
		}
	}

	
	public static Configuration getMainConfiguration() {
		return mainConfig;
	}
	
	public static Configuration setMainConfiguration(Configuration conf) {
		return mainConfig = conf;
	}

}
