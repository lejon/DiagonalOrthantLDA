package xyz.lejon.utils;

import static org.ejml.ops.CommonOps.multTransA;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import joinery.DataFrame;
import joinery.DataFrame.NumberDefault;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.ejml.data.DenseMatrix64F;
import org.junit.Assert;
import org.junit.Test;

import xyz.lejon.sampling.FixedCovarianceMultivariateNormalDistribution;

public class MatrixOpTest {
	
	@Test
	public void testDot() {
		double [] a = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0};
		double [] b = {9.0,8.0,7.0,6.0,5.0,4.0,3.0,2.0};
		assertEquals(156.0, MatrixOps.dot(a,b), 0.00000001);
		System.out.println(MatrixOps.dot(a,b));
	}

	@Test
	public void testDot2P1() {
		double [] a = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0};
		double [] a1 = {1.0,2.0,3.0,4.0,5.0};
		double [] a2 = {6.0,7.0,8.0};
		double [] b = {9.0,8.0,7.0,6.0,5.0,4.0,3.0,2.0};
		assertEquals(MatrixOps.dot(a,b), MatrixOps.dot2P1(a1,a2,b), 0.00000001);
	}

	@Test
	public void testDotvsR() throws IOException {
		DataFrame<Object> df = DataFrame.readCsv("src/main/resources/datasets/iris-small.csv",",",NumberDefault.DOUBLE_DEFAULT,"NA", false);

		int noRows = df.length();

		Map<String,Integer> labels = new HashMap<>();
		int [] ys = new int[noRows];
		int idx = 0;
		int lblcnt = 0;
		for (Object lbl : df.col(4)) {
			String label = lbl.toString();
			if(labels.get(label)==null) {
				labels.put(label,lblcnt++);
			}
			ys[idx++] = labels.get(label);
		}

		// Now drop the label column
		df = df.drop(4);
		
		DataFrame<Double> ddf = df.cast(Double.class);
		double [][] xs = ddf.fillna(0.0).toArray(double[][].class);
		int noClasses = labels.size();
		double [][] expected = {
				{-49.10359,-18.35522,-46.26905, -18.20017}, 
				{-32.01776,-20.52999,-15.78441, -6.31844},
				{-51.01041,-31.63687, -18.75323,  -2.22077}}; 
		
		double [][] Zs = {
				{0.1419, -1.7705, -0.0195},
				{0.2448, -0.0313, -2.9330},
				{0.6627, -0.3274, -1.1980},
				{1.3047, -0.0671, -2.3800},
				{0.2491, -1.1890, -0.6032},
				{0.8545, -0.6127, -0.0029},
				{0.9993, -0.8015, -0.6134},
				{-0.3840, 0.4126, -0.5430},
				{-1.3735, 0.4659, -0.6960},
				{-0.5706, 1.3060, -0.6848},
				{-0.2725, 0.2894, -1.4065},
				{-2.1009, 0.3854, -0.4060},
				{-1.0277, 0.0011, -0.9705},
				{-0.6492, 0.4990, -1.2531},
				{-0.9672, -1.2763, 0.7156},
				{-1.9490, -0.2691, 1.3374},
				{-0.2459, -0.7731, 0.0967},
				{-1.8045, -0.4012, 0.1637},
				{-0.1097, -0.9174, 0.5803},
				{-0.1217, -0.5986, 0.0015}};

		DenseMatrix64F Xd        = new DenseMatrix64F(xs);
		for (int classIdx = 0; classIdx < noClasses; classIdx++) {
			double [][] zCol = MatrixOps.extractCol(classIdx,Zs);

			DenseMatrix64F zCold = new DenseMatrix64F(zCol);
			DenseMatrix64F mu = new DenseMatrix64F(Xd.numCols, zCold.numCols);

			multTransA(Xd, zCold, mu);

			double [] mu_tile = MatrixOps.transposeSerial(MatrixOps.extractDoubleArray(mu))[0];
			System.out.println(MatrixOps.arrToStr(mu_tile, "Mean"));
			assertArrayEquals(expected[classIdx], mu_tile, 0.2);
		}

	}

	@Test
	public void testNormalCDF() {
		NormalDistribution nd = new NormalDistribution();
		assertEquals(0.7115, nd.cumulativeProbability(0.5578), 0.00001);
		System.out.println(nd.cumulativeProbability(0.5578));
		assertEquals(0.2885, nd.cumulativeProbability(-0.5578), 0.00001);
		System.out.println(nd.cumulativeProbability(-0.5578));
	}
	
	@Test
	public void testPCATransform() {
		FixedCovarianceMultivariateNormalDistribution myfcmvn;
		double [] mu = {0.0,0.0,0.0,0.0,0.0};
		double [][] S  = {
				{1.0,0.0,0.0,0.0,0.0},
				{0.0,1.0,0.0,0.0,0.0},
				{0.0,0.0,1.0,0.0,0.0},
				{0.0,0.0,0.0,1.0,0.0},
				{0.0,0.0,0.0,0.0,1.0},
		};
		myfcmvn = new FixedCovarianceMultivariateNormalDistribution(mu,S);
		
		int noRows = 100;
		int noCols = 5;
		double [][] betas = new double[noRows][noCols]; 
		
		for (int row = 0; row < noRows; row++) {
			betas[row] = myfcmvn.sample(mu);			
		}
		System.out.println(MatrixOps.doubleArrayToPrintString(betas));
		
		PrincipalComponentAnalysis pca = new PrincipalComponentAnalysis();
		double [][] pcaBetas = pca.pca(betas, 3);
		
		System.out.println(MatrixOps.doubleArrayToPrintString(pcaBetas));
		
		double [][] pcaBetasAgain = pca.translateToSpace(betas);
		
		System.out.println(MatrixOps.doubleArrayToPrintString(pcaBetasAgain));
		assertArrayEquals(pcaBetas, pcaBetasAgain);
		
		double [][] pcaBetasBack = pca.translateFromSpace(pcaBetas);
		System.out.println("To PCA and back");
		System.out.println(MatrixOps.doubleArrayToPrintString(pcaBetasBack,10,10));
		System.out.println(MatrixOps.doubleArrayToPrintString(betas,10,10));
		
		System.out.println();
		System.out.println(MatrixOps.arrToStr(betas[0], "Sample"));
		double [] eigs = pca.sampleToEigenSpace(betas[0]);
		System.out.println(MatrixOps.arrToStr(eigs, "Eigen"));
		System.out.println(MatrixOps.arrToStr(pca.eigenToSampleSpace(eigs), "Sample Back"));
	}
	
	@Test
	public void testConcatMatrices() {
		double [][] m1  = {
				{1.0,0.0,0.0,0.0,0.1},
				{1.0,1.0,0.0,0.0,0.1},
				{1.0,0.0,1.0,0.0,0.1},
				{1.0,0.0,0.0,1.0,0.1},
				{1.0,0.0,0.0,0.0,1.1},
		};
		double [][] m2  = {
				{2.0,0.0,0.0,0.0,0.2},
				{2.0,2.0,0.0,0.0,0.2},
				{2.0,0.0,2.0,0.0,0.2},
				{2.0,0.0,0.0,2.0,0.2},
				{2.0,0.0,0.0,0.0,2.2},
		};
		double [][] expected  = {
				{1.0,0.0,0.0,0.0,0.1, 2.0,0.0,0.0,0.0,0.2},
				{1.0,1.0,0.0,0.0,0.1, 2.0,2.0,0.0,0.0,0.2},
				{1.0,0.0,1.0,0.0,0.1, 2.0,0.0,2.0,0.0,0.2},
				{1.0,0.0,0.0,1.0,0.1, 2.0,0.0,0.0,2.0,0.2},
				{1.0,0.0,0.0,0.0,1.1, 2.0,0.0,0.0,0.0,2.2},
		};
		
		double [][] conct = MatrixOps.concatenate(m1, m2);
		Assert.assertArrayEquals( expected, conct );
	}
	
	@Test
	public void testConcatEmptyMatrix() {
		double [][] m1  = {
				{1.0,0.0,0.0,0.0,0.1},
				{1.0,1.0,0.0,0.0,0.1},
				{1.0,0.0,1.0,0.0,0.1},
				{1.0,0.0,0.0,1.0,0.1},
				{1.0,0.0,0.0,0.0,1.1},
		};
		double [][] m2  = {
				{},
				{},
				{},
				{},
				{},
		};
		double [][] expected  = {
				{1.0,0.0,0.0,0.0,0.1},
				{1.0,1.0,0.0,0.0,0.1},
				{1.0,0.0,1.0,0.0,0.1},
				{1.0,0.0,0.0,1.0,0.1},
				{1.0,0.0,0.0,0.0,1.1},
		};
		
		double [][] conct = MatrixOps.concatenate(m1, m2);
		Assert.assertArrayEquals( expected, conct );
	}
	
	@Test
	public void testConcatEmptyMatrixFirst() {
		double [][] m1  = {
				{1.0,0.0,0.0,0.0,0.1},
				{1.0,1.0,0.0,0.0,0.1},
				{1.0,0.0,1.0,0.0,0.1},
				{1.0,0.0,0.0,1.0,0.1},
				{1.0,0.0,0.0,0.0,1.1},
		};
		double [][] m2  = {
				{},
				{},
				{},
				{},
				{},
		};
		double [][] expected  = {
				{1.0,0.0,0.0,0.0,0.1},
				{1.0,1.0,0.0,0.0,0.1},
				{1.0,0.0,1.0,0.0,0.1},
				{1.0,0.0,0.0,1.0,0.1},
				{1.0,0.0,0.0,0.0,1.1},
		};
		
		double [][] conct = MatrixOps.concatenate(m2,m1);
		Assert.assertArrayEquals( expected, conct );
	}

	
	@Test
	public void testConcatMatrixVector() {
		double [][] m1  = {
				{1.0,0.0,0.0,0.0,0.1},
				{1.0,1.0,0.0,0.0,0.1},
				{1.0,0.0,1.0,0.0,0.1},
				{1.0,0.0,0.0,1.0,0.1},
				{1.0,0.0,0.0,0.0,1.1},
		};
		double [] v2  = {2.0,0.0,0.0,0.0,2.0};
		double [][] expected  = {
				{1.0,0.0,0.0,0.0,0.1, 2.0},
				{1.0,1.0,0.0,0.0,0.1, 0.0},
				{1.0,0.0,1.0,0.0,0.1, 0.0},
				{1.0,0.0,0.0,1.0,0.1, 0.0},
				{1.0,0.0,0.0,0.0,1.1, 2.0},
		};
		
		double [][] conct = MatrixOps.concatenate(m1, v2);
		Assert.assertArrayEquals( expected, conct );
	}
	
	@Test
	public void testConcatVectorVector() {
		double [] v1  = {1.0,0.0,0.0,0.0,1.0};
		double [] v2  = {2.0,0.0,0.0,0.0,2.0};
		double [] expected  = {1.0,0.0,0.0,0.0,1.0,2.0,0.0,0.0,0.0,2.0};
		
		double [] conct = MatrixOps.concatenate(v1, v2);
		Assert.assertArrayEquals( expected, conct, 0.00000001 );
	}

	@Test
	public void testExtractSubMatrix() {
		double [][] m1  = {
				{1.0,0.0,0.0,0.0,0.1},
				{1.0,1.0,2.0,0.0,0.1},
				{1.0,3.0,4.0,0.0,0.1},
				{1.0,0.0,0.0,1.0,0.1},
				{1.0,0.0,0.0,0.0,1.1},
		};
		int [] cols = {1,2};
		int [] rows = {1,2};
		
		double [][] expected  = {
				{1.0,2.0},
				{3.0,4.0},
		};
		
		double [][] extr = MatrixOps.extractSubMatrix(cols, rows, m1);
		Assert.assertArrayEquals( expected, extr );
	}	
	
	@Test
	public void testExtractSubMatrixUpperLeft() {
		double [][] m1  = {
				{1.0,2.0,0.0,0.0,0.1},
				{3.0,4.0,0.0,0.0,0.1},
				{1.0,0.0,1.0,0.0,0.1},
				{1.0,0.0,0.0,1.0,0.1},
				{1.0,0.0,0.0,0.0,1.1},
		};
		int [] cols = {0,1};
		int [] rows = {0,1};
		
		double [][] expected  = {
				{1.0,2.0},
				{3.0,4.0},
		};
		
		double [][] extr = MatrixOps.extractSubMatrix(cols, rows, m1);
		Assert.assertArrayEquals( expected, extr );
	}	
	
	@Test
	public void testExtractSubMatrixLeftPart() {
		double [][] m1  = {
				{1.0,2.0,0.0,0.0,0.1},
				{3.0,4.0,0.0,0.0,0.1},
				{5.0,6.0,1.0,0.0,0.1},
				{7.0,8.0,0.0,1.0,0.1},
				{9.0,10.0,0.0,0.0,1.1},
		};
		int [] cols = {0,1};
		int [] rows = {0,1,2,3,4};
		
		double [][] expected  = {
				{1.0,2.0},
				{3.0,4.0},
				{5.0,6.0},
				{7.0,8.0},
				{9.0,10.0}
		};
		
		double [][] extr = MatrixOps.extractSubMatrix(cols, rows, m1);
		Assert.assertArrayEquals( expected, extr );
	}	
	
	@Test
	public void testExtractSubMatrixRightPart() {
		double [][] m1  = {
				{0.0,0.0,0.1,1.0,2.0},
				{0.0,0.0,0.1,3.0,4.0},
				{1.0,0.0,0.1,5.0,6.0},
				{0.0,1.0,0.1,7.0,8.0},
				{0.0,0.0,1.1,9.0,10.0},
		};
		int [] cols = {3,4};
		int [] rows = {0,1,2,3,4};
		
		double [][] expected  = {
				{1.0,2.0},
				{3.0,4.0},
				{5.0,6.0},
				{7.0,8.0},
				{9.0,10.0}
		};
		
		double [][] extr = MatrixOps.extractSubMatrix(cols, rows, m1);
		Assert.assertArrayEquals( expected, extr );
	}	
	
	@Test
	public void testExtractEmpty() {
		double [][] m1  = {
				{0.0,0.0,0.1,1.0,2.0},
				{0.0,0.0,0.1,3.0,4.0},
				{1.0,0.0,0.1,5.0,6.0},
				{0.0,1.0,0.1,7.0,8.0},
				{0.0,0.0,1.1,9.0,10.0},
		};
		int [] cols = {};
		int [] rows = {};
		
		double [][] expected  = {{}};
		
		double [][] extr = MatrixOps.extractSubMatrix(cols, rows, m1);
		Assert.assertArrayEquals( expected, extr );
	}	

	@Test
	public void testExtractColsArray() {
		double [][] m1  = {
				{0.0,0.0,0.1,1.0,2.0},
				{0.0,0.0,0.1,3.0,4.0},
				{1.0,0.0,0.1,5.0,6.0},
				{0.0,1.0,0.1,7.0,8.0},
				{0.0,0.0,1.1,9.0,10.0},
		};
		int [] cols = {2,3};
		
		double [][] expected  = {
				{0.1,1.0},
				{0.1,3.0},
				{0.1,5.0},
				{0.1,7.0},
				{1.1,9.0},
		};
		
		double [][] extr = MatrixOps.extractCols(cols, m1);
		Assert.assertArrayEquals( expected, extr );
	}	
	
	@Test
	public void testExtractColsIdxs() {
		double [][] m1  = {
				{0.0,0.0,0.1,1.0,2.0},
				{0.0,0.0,0.1,3.0,4.0},
				{1.0,0.0,0.1,5.0,6.0},
				{0.0,1.0,0.1,7.0,8.0},
				{0.0,0.0,1.1,9.0,10.0},
		};
		
		double [][] expected  = {
				{0.1,1.0},
				{0.1,3.0},
				{0.1,5.0},
				{0.1,7.0},
				{1.1,9.0},
		};
		
		double [][] extr = MatrixOps.extractCols(2,4, m1);
		Assert.assertArrayEquals( expected, extr );
	}	


}
