package xyz.lejon.runnables;

import java.io.PrintWriter;

import org.apache.commons.cli.ParseException;
import org.apache.commons.configuration.ConfigurationException;

import xyz.lejon.configuration.DOLDAConfiguration;
import xyz.lejon.utils.LoggingUtils;
import cc.mallet.configuration.ConfigFactory;
import cc.mallet.configuration.LDACommandLineParser;
import cc.mallet.configuration.ParsedLDAConfiguration;
import cc.mallet.topics.HierarchicalLDA;
import cc.mallet.util.LDAUtils;
import cc.mallet.types.InstanceList;
import cc.mallet.util.Randoms;

public class HLDA {


	public static void main (String[] args) throws java.io.IOException, ParseException, ConfigurationException {

		System.out.println("We have: " + Runtime.getRuntime().availableProcessors() 
				+ " processors avaiable");
		String buildVer = LoggingUtils.getManifestInfo("Implementation-Build","DOLDA");
		String implVer  = LoggingUtils.getManifestInfo("Implementation-Version", "DOLDA");
		if(buildVer==null||implVer==null) {
			System.out.println("GIT info:" + LoggingUtils.getLatestCommit());
		} else {
			System.out.println("Build info:" 
					+ "Implementation-Build = " + buildVer + ", " 
					+ "Implementation-Version = " + implVer);
		}
		
		LDACommandLineParser cp = new LDACommandLineParser(args);
		
		// We have to create this temporary config because at this stage if we want to create a new config for each run
		ParsedLDAConfiguration config = (ParsedLDAConfiguration) ConfigFactory.getMainConfiguration(cp);

		String [] configs = config.getSubConfigs();
		config.activateSubconfig(configs[0]);
		
		String textdataset_fn = config.getDatasetFilename();
		System.out.println("Filename is: " + textdataset_fn);
		String stoplistFn = config.getStringProperty("stoplist");
		
		InstanceList instances = LDAUtils.loadInstances(textdataset_fn, 
				stoplistFn, config.getRareThreshold(DOLDAConfiguration.RARE_WORD_THRESHOLD));

		HierarchicalLDA hlda = new HierarchicalLDA();

		// Set hyperparameters

		double alpha = 10.0; 
		hlda.setAlpha(alpha);
		double gamma = 1.0;
		hlda.setGamma(gamma);
		double eta =  0.1;
		hlda.setEta(eta);

		// Display preferences

		hlda.setTopicDisplay(10, 15);
		hlda.setProgressDisplay(true);

		// Initialize random number generator

		Randoms random = new Randoms();

		// Initialize and start the sampler

		hlda.initialize(instances, null, 3, random);
		hlda.estimate(1000);

		// Output results

		String stateFile = "hlda-state";
		hlda.printState(new PrintWriter(stateFile));
	}
}
