package xyz.lejon.sampling;

import static org.junit.Assert.fail;

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
public class SolveParTest {
	
	int noLoops = 1000;
	int p = 100;
		
	class EJMLSolver extends RecursiveAction {
		final static long serialVersionUID = 1L;
		int startIdx = -1;
		int endIdx = -1;
		double [][] store;
		int limit = 10;
		double [] means;
		double[][] cov;
		DenseMatrix64F localMu;
		DenseMatrix64F dCov;
		DenseMatrix64F dMeans;
		
		
		public EJMLSolver(int startClass, int endClass, double [][]  store, double [] means, double [][] cov) {
			this.store = store;
			this.startIdx = startClass;
			this.endIdx = endClass;
			this.means = means;
			this.cov = cov;
			
			
			localMu = new DenseMatrix64F(means.length,1);
			dMeans = new DenseMatrix64F(means.length,1);
			dMeans.setData(means);
			dCov = new DenseMatrix64F(cov);
		}

		@Override
		protected void compute() {
			if ( (endIdx-startIdx) <= limit ) {
				//System.out.println("Solving between " + startIdx + " and " + endIdx);
				for (int classNo = startIdx; classNo < endIdx; classNo++) {
					org.ejml.ops.CommonOps.solve(dCov,dMeans,localMu);
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
				invokeAll(new EJMLSolver(startClass1, endClass1, store, means, cov),
						new EJMLSolver(startClass2, endClass2, store, means, cov));
			}
		}
	}
	
	class EJMLInverter extends RecursiveAction {
		final static long serialVersionUID = 1L;
		int startIdx = -1;
		int endIdx = -1;
		double [][] store;
		int limit = 10;
		double [] means;
		double[][] cov;
		DenseMatrix64F localMu;
		DenseMatrix64F dCov;
		DenseMatrix64F dMeans;
		
		
		public EJMLInverter(int startClass, int endClass, double [][]  store, double [] means, double [][] cov) {
			this.store = store;
			this.startIdx = startClass;
			this.endIdx = endClass;
			this.means = means;
			this.cov = cov;
			localMu = new DenseMatrix64F(means.length);
			dMeans = new DenseMatrix64F(means.length);
			dMeans.setData(means);
			dCov = new DenseMatrix64F(cov);
		}

		@Override
		protected void compute() {
			if ( (endIdx-startIdx) <= limit ) {
				for (int classNo = startIdx; classNo < endIdx; classNo++) {
					org.ejml.ops.CommonOps.invert(dCov);
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
				invokeAll(new EJMLInverter(startClass1, endClass1, store, means, cov),
						new EJMLInverter(startClass2, endClass2, store, means, cov));
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
		
		EJMLSolver mts = new EJMLSolver(0, noLoops, samples, means, cov);
		
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
		double [][] cov = MatrixOps.extractDoubleArray(covd2);
		
		EJMLInverter mts = new EJMLInverter(0, noLoops, samples, means, cov);
		
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
