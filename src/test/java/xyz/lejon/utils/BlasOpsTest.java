package xyz.lejon.utils;

import static org.junit.Assert.*;

import org.jblas.DoubleMatrix;
import org.jblas.Solve;
import org.junit.Test;

public class BlasOpsTest {

	@Test
	public void testInvert() {
		double [][] a = {{3,2},{1,1}};
		//System.out.println("a="+ MatrixOps.doubleArrayToPrintString(a));
		double[][] b = {{8},{2}};
		//System.out.println("b="+ MatrixOps.doubleArrayToPrintString(b));
		
		DoubleMatrix ad = new DoubleMatrix(a);
		DoubleMatrix bd = new DoubleMatrix(b);
		
		DoubleMatrix sol1 = Solve.solve(ad, bd);
		//System.out.println("Sol 1 =" + sol1);
		assertEquals(4, sol1.get(0),0.00000001);
		assertEquals(-2, sol1.get(1),0.00000001);
		
		DoubleMatrix adInv = BlasOps.blasInvert(ad);
		DoubleMatrix sol2 = adInv.mmul(bd);
		//System.out.println("Sol 2 =" + sol2);
		assertEquals(4, sol2.get(0),0.00000001);
		assertEquals(-2, sol2.get(1),0.00000001);
	}

}
