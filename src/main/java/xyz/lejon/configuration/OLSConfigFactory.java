package xyz.lejon.configuration;

import org.apache.commons.configuration.ConfigurationException;

public class OLSConfigFactory {
	protected static Configuration mainConfig = null;
	public static Configuration getMainConfiguration(OLSCommandLineParser cp) throws ConfigurationException {
		if( mainConfig == null ) {
			mainConfig = new ParsedOLSConfiguration(cp);
		}
		
		return mainConfig;
	}

	public static Configuration getMainConfiguration() {
		return mainConfig;
	}
	
	public static Configuration setMainConfiguration(Configuration conf) {
		return mainConfig = conf;
	}

}
