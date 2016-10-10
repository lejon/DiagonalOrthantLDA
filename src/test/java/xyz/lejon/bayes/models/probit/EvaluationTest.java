package xyz.lejon.bayes.models.probit;

import static org.junit.Assert.fail;

import java.util.HashMap;
import java.util.Map;

import org.junit.Assert;
import org.junit.Test;

public class EvaluationTest {

	@Test
	public void testArgs() {
		int [] predicted = {0,1};
		int [] real      = {0};
		try {
			DOEvaluation.buildConfusionMatrix(real, predicted, 2);
			fail("Should have thrown IllegalArgumentException since we have non-matching predicted and real");
		} catch (IllegalArgumentException e) {
		}
	}
	
	@Test
	public void testBuildConfusionMatrixBinaryBasic() {
		Map<Integer,String> idMap = new HashMap<>();
		idMap.put(0, "false");
		idMap.put(1, "true");
		int [] predicted = {0,1};
		int [] real      = {0,1};
		int [][] expectedConfusion = {{1,0},{0,1}};
		int [][] builtConfusion = DOEvaluation.buildConfusionMatrix(real, predicted, 2);
		Assert.assertArrayEquals( expectedConfusion, builtConfusion );
		System.out.println(DOEvaluation.confusionMatrixToString(builtConfusion, idMap));
	}

	@Test
	public void testBuildConfusionMatrixBinaryMis() {
		int [] predicted = {0,1};
		int [] real      = {1,0};
		int [][] expectedConfusion = {{0,1},{1,0}};
		int [][] builtConfusion = DOEvaluation.buildConfusionMatrix(real, predicted, 2);
		Assert.assertArrayEquals( expectedConfusion, builtConfusion );
	}

	@Test
	public void testBuildConfusionMatrixBinary2Mis() {
		int [] predicted = {0,1,0,1};
		int [] real      = {1,0,1,0};
		int [][] expectedConfusion = {{0,2},{2,0}};
		int [][] builtConfusion = DOEvaluation.buildConfusionMatrix(real, predicted, 2);
		Assert.assertArrayEquals( expectedConfusion, builtConfusion );
	}

	@Test
	public void testBuildConfusionMatrixBinary2() {
		int [] predicted = {0,1,0,1};
		int [] real      = {0,1,0,1};
		int [][] expectedConfusion = {{2,0},{0,2}};
		int [][] builtConfusion = DOEvaluation.buildConfusionMatrix(real, predicted, 2);
		Assert.assertArrayEquals( expectedConfusion, builtConfusion );
	}
	
	@Test
	public void testBuildConfusionMatrixBinaryEven() {
		int [] predicted = {1,0,0,1};
		int [] real      = {0,1,0,1};
		int [][] expectedConfusion = {{1,1},{1,1}};
		int [][] builtConfusion = DOEvaluation.buildConfusionMatrix(real, predicted, 2);
		Assert.assertArrayEquals( expectedConfusion, builtConfusion );
	}

	@Test
	public void testBuildConfusionMatrixTertiary() {
		int [] predicted = {0,1,2};
		int [] real      = {0,1,2};
		int [][] expectedConfusion = {{1,0,0},{0,1,0},{0,0,1}};
		int [][] builtConfusion = DOEvaluation.buildConfusionMatrix(real, predicted, 3);
		Assert.assertArrayEquals( expectedConfusion, builtConfusion );
	}
	
	@Test
	public void testBuildConfusionMatrixTertiary2() {
		Map<Integer,String> idMap = new HashMap<>();
		idMap.put(0, "yes");
		idMap.put(1, "no");
		idMap.put(2, "maybe");
		int [] predicted = {0,1,2,0,1,2};
		int [] real      = {0,1,2,0,1,2};
		int [][] expectedConfusion = {{2,0,0},{0,2,0},{0,0,2}};
		int [][] builtConfusion = DOEvaluation.buildConfusionMatrix(real, predicted, 3);
		Assert.assertArrayEquals( expectedConfusion, builtConfusion );
		System.out.println(DOEvaluation.confusionMatrixToString(builtConfusion, idMap));
	}
	
	@Test
	public void testBuildConfusionMatrixTertiaryMis() {
		int [] predicted = {0,1,2};
		int [] real      = {2,0,1};
		int [][] expectedConfusion = {{0,0,1},{1,0,0},{0,1,0}};
		int [][] builtConfusion = DOEvaluation.buildConfusionMatrix(real, predicted, 3);
		Assert.assertArrayEquals( expectedConfusion, builtConfusion );
	}
	
	@Test
	public void testBuildConfusionMatrixTertiaryEven() {
		Map<Integer,String> idMap = new HashMap<>();
		idMap.put(0, "no");
		idMap.put(1, "yes");
		idMap.put(2, "maybe");
		int [] predicted = {0,1,2,0,1,2,0,1,2};
		int [] real      = {0,1,2,1,0,0,2,2,1};
		int [][] expectedConfusion = {{1,1,1},{1,1,1},{1,1,1}};
		int [][] builtConfusion = DOEvaluation.buildConfusionMatrix(real, predicted, 3);
		Assert.assertArrayEquals( expectedConfusion, builtConfusion );
		System.out.println(DOEvaluation.confusionMatrixToString(builtConfusion, idMap));
	}
	
	@Test
	public void testBuildConfusionMatrixTertiaryPyramid() {
		Map<Integer,String> idMap = new HashMap<>();
		idMap.put(0, "no");
		idMap.put(1, "yes");
		idMap.put(2, "maybe");
		int [] predicted = {0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,2,2};
		int [] real      = {0,1,1,2,2,2,0,0,0,1,2,2,0,0,0,0,1,1,1,2};
		int [][] expectedConfusion = {
				{1,2,3},
				{3,1,2},
				{4,3,1}};
		int [][] builtConfusion = DOEvaluation.buildConfusionMatrix(real, predicted, 3);
		Assert.assertArrayEquals( expectedConfusion, builtConfusion );
		System.out.println(DOEvaluation.confusionMatrixToString(builtConfusion, idMap));
	}
}
