package xyz.lejon.sampling;

import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.RandomMatrices;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import xyz.lejon.MarkerIFParallelTest;
import xyz.lejon.utils.MatrixOps;

@Category(MarkerIFParallelTest.class)
public class SolveSharedParTest {
	
	int noLoops = 1000;
	int p = 500;
		
	class EJMLSolver extends RecursiveAction {
		final static long serialVersionUID = 1L;
		int startIdx = -1;
		int endIdx = -1;
		double [][] store;
		int limit = 10;
		DenseMatrix64F localMu;
		DenseMatrix64F dCov;
		DenseMatrix64F dMeans;
		
		
		public EJMLSolver(int startClass, int endClass, double [][]  store, DenseMatrix64F dMeans, DenseMatrix64F dCov) {
			this.store = store;
			this.startIdx = startClass;
			this.endIdx = endClass;
			this.dMeans = dMeans;
			this.dCov = dCov;
			localMu = new DenseMatrix64F(dMeans.numRows,1);
		}

		@Override
		protected void compute() {
			if ( (endIdx-startIdx) <= limit ) {
				//System.out.println("Solving between " + startIdx + " and " + endIdx);
				for (int classNo = startIdx; classNo < endIdx; classNo++) {
					assertTrue(org.ejml.ops.CommonOps.solve(dCov,dMeans,localMu));
					if(containsNaNs(MatrixOps.extractDoubleArray(dCov))) {
						fail("Solving NaN's");
					}
				}
			}
			else {
				int range = (endIdx-startIdx);
				int startClass1 = startIdx;
				int endClass1 = startIdx + (range / 2);
				int startClass2 = endClass1;
				int endClass2 = endIdx;
				invokeAll(new EJMLSolver(startClass1, endClass1, store, dMeans, dCov),
						new EJMLSolver(startClass2, endClass2, store, dMeans, dCov));
			}
		}
	}
	
	class EJMLInverter extends RecursiveAction {
		final static long serialVersionUID = 1L;
		int startIdx = -1;
		int endIdx = -1;
		double [][] store;
		int limit = 10;
		DenseMatrix64F localMu;
		DenseMatrix64F dCov;
		DenseMatrix64F dMeans;
		
		
		public EJMLInverter(int startClass, int endClass, double [][]  store, DenseMatrix64F dMeans, DenseMatrix64F dCov) {
			this.store = store;
			this.startIdx = startClass;
			this.endIdx = endClass;
			this.dMeans = dMeans;
			this.dCov = dCov;
			localMu = new DenseMatrix64F(dMeans.numRows,1);
		}

		@Override
		protected void compute() {
			if ( (endIdx-startIdx) <= limit ) {
				//System.out.println("Inverting between " + startIdx + " and " + endIdx);
				for (int classNo = startIdx; classNo < endIdx; classNo++) {
					assertTrue(org.ejml.ops.CommonOps.invert(dCov));
					if(containsNaNs(MatrixOps.extractDoubleArray(dCov))) {
						fail("Inverted NaN's");
					}
				}
			}
			else {
				int range = (endIdx-startIdx);
				int startClass1 = startIdx;
				int endClass1 = startIdx + (range / 2);
				int startClass2 = endClass1;
				int endClass2 = endIdx;
				invokeAll(new EJMLInverter(startClass1, endClass1, store, dMeans, dCov),
						new EJMLInverter(startClass2, endClass2, store, dMeans, dCov));
			}
		}

	}

	private boolean containsNaNs(double[][] ds) {
		for (int i = 0; i < ds.length; i++) {
			for (int j = 0; j < ds[i].length; j++) {
				if(Double.isInfinite(ds[i][j]) || Double.isNaN(ds[i][j])) {
					return true; 
				}
			}
		}
		return false;
	}

	@Test
	public void testParEJMLSolve() {
		double [][] samples = new double[noLoops][]; 
		long start = System.currentTimeMillis();
		
		double [] means= new double[p]; 
		Arrays.fill(means, 0.0);
		System.out.println("Mean dim: 1 x " + means.length);
		System.out.println("Mean is at: " + means[0] + "x" + means[1]);

		Random rand = new Random();
		DenseMatrix64F covd2 = RandomMatrices.createSymmPosDef(p, rand);
		double [][] cov = MatrixOps.extractDoubleArray(covd2);
		
		DenseMatrix64F dMeans = new DenseMatrix64F(means.length,1);
		dMeans.setData(means);
		DenseMatrix64F dCov = new DenseMatrix64F(cov);

		EJMLSolver mts = new EJMLSolver(0, noLoops, samples, dMeans, dCov);
		
		ForkJoinPool samplerPool = new ForkJoinPool(Runtime.getRuntime().availableProcessors(), ForkJoinPool.defaultForkJoinWorkerThreadFactory, null, true);
		try {
			samplerPool.invoke(mts);
		} catch (Exception e) {
			e.printStackTrace();
			fail("Exception: " + e);
		}
		System.out.println("Sampling took: " + (System.currentTimeMillis() - start) + " milliseconds...");
	}
	
	@Test
	public void testParEJMLInv() {
		double [][] samples = new double[noLoops][]; 
		long start = System.currentTimeMillis();
		
		double [] means= new double[p]; 
		Arrays.fill(means, 0.0);
		System.out.println("Mean dim: 1 x " + means.length);
		System.out.println("Mean is at: " + means[0] + "x" + means[1]);

		Random rand = new Random();
		DenseMatrix64F covd2 = RandomMatrices.createSymmPosDef(p, rand);
		
		DenseMatrix64F dMeans = new DenseMatrix64F(means.length,1);
		dMeans.setData(means);
		
		EJMLInverter mts = new EJMLInverter(0, noLoops, samples, dMeans, covd2);
		
		ForkJoinPool samplerPool = new ForkJoinPool(Runtime.getRuntime().availableProcessors(), ForkJoinPool.defaultForkJoinWorkerThreadFactory, null, true);
		try {
			samplerPool.invoke(mts);
		} catch (Exception e) {
			e.printStackTrace();
			fail("Exception: " + e);
		}
		System.out.println("Sampling took: " + (System.currentTimeMillis() - start) + " milliseconds...");
	}
	
	
}
