package xyz.lejon.utils;

import static org.junit.Assert.*;

import org.jblas.DoubleMatrix;
import org.junit.Test;

public class DoubleMatrixOwnershipTest {

	@Test
	public void test() {
		double [][] Xs = {{1,2,3},{4,5,6}};
		DoubleMatrix X = new DoubleMatrix(Xs);
		X.put(0, 75);
		assertTrue("DoubleMatrix doesnt copy input anymore!", Xs[0][0]!=X.get(0,0));
	}

}
