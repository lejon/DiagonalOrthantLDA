package xyz.lejon.bayes.models.dolda;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;

import org.apache.commons.cli.ParseException;
import org.apache.commons.configuration.ConfigurationException;
import org.junit.Test;

import xyz.lejon.configuration.DOLDACommandLineParser;
import xyz.lejon.configuration.DOLDAConfiguration;
import xyz.lejon.configuration.ParsedDOLDAConfiguration;
import cc.mallet.types.InstanceList;

public class FIxedCrossValidationIteratorTest {

	@Test
	public void test() throws ParseException, ConfigurationException, IOException {
		int TRAINING = 0;
		int TESTING = 1;

		String [] args = {"--run_cfg=src/main/resources/configuration/DOLDABasicTest.cfg"};
		DOLDACommandLineParser cp = new DOLDACommandLineParser(args);
		DOLDAConfiguration config = new ParsedDOLDAConfiguration(cp);
		config.forceActivateSubconfig("DOLDA-films-imdb");
		
		DOLDADataSet trainingSetData = config.loadCombinedTrainingSet();
		
		InstanceList textData = trainingSetData.getTextData();
		
		// Some ID's are missing in the dataset
		List<String>  fixedTrainingIds = Arrays.asList(new String[]{"movie-0", "movie-1", "movie-2", "movie-3", "movie-4", "movie-6", "movie-7", "movie-8", "movie-9", "movie-12"});
		FixedCrossValidationIterator iter = new FixedCrossValidationIterator(textData, 1, new Random(System.currentTimeMillis()), fixedTrainingIds);
		InstanceList[] set = iter.next();
		assertEquals(textData.size()-fixedTrainingIds.size(),set[TESTING].size());
		assertEquals(fixedTrainingIds.size(),set[TRAINING].size());
		try {
			iter.next();
			fail("Expected NoSuchElementException!");
		} catch (NoSuchElementException e) {
		}
		for (int i = 0; i < set[TRAINING].size(); i++) {
			assertTrue(set[TRAINING].get(i).getName() + " was not in trainingset", fixedTrainingIds.contains(set[TRAINING].get(i).getName()));
		}
		for (int i = 0; i < set[TESTING].size(); i++) {
			assertTrue(set[TESTING].get(i).getName() + " WAS in trainingset", !fixedTrainingIds.contains(set[TESTING].get(i).getName()));
		}
		
	}

}
