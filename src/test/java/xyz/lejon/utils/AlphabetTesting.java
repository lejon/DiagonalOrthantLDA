package xyz.lejon.utils;

import static org.junit.Assert.*;

import java.io.FileNotFoundException;

import org.junit.Test;

import cc.mallet.util.LDAUtils;
import cc.mallet.types.InstanceList;
import cc.mallet.util.Randoms;

public class AlphabetTesting {

	@Test
	public void test() throws FileNotFoundException {
		int TRAINING = 0;
		int TESTING = 1;

		// Load textdata
		InstanceList instances = LDAUtils.loadInstances("src/test/resources/datasets/small.lda", null, 1);
		InstanceList[] instanceLists =
				instances.split(new Randoms(),
						new double[] {0.9, 0.1, 0.0});
		
		//System.out.println(instanceLists[TESTING].getAlphabet());
		//System.out.println(instanceLists[TRAINING].getAlphabet());
		
		assertEquals(instanceLists[TRAINING].getAlphabet(),instanceLists[TESTING].getAlphabet());
	}

}
