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
public class MVNSamplerTest {
	
	int noLoops = 100;
	int p = 500;
		
	class EJMLSampler extends RecursiveAction {
		final static long serialVersionUID = 1L;
		int startIdx = -1;
		int endIdx = -1;
		double [][] store;
		int limit = 10;
		double [] means;
		double[][] cov;
		
		public EJMLSampler(int startClass, int endClass, double [][]  store, double [] means, double [][] cov) {
			this.store = store;
			this.startIdx = startClass;
			this.endIdx = endClass;
			this.means = means;
			this.cov = cov;
		}

		@Override
		protected void compute() {
			if ( (endIdx-startIdx) <= limit ) {
				for (int classNo = startIdx; classNo < endIdx; classNo++) {
					FastMVNSamplerEJML sampler = new FastMVNSamplerEJML(means, cov);
					store[classNo] = sampler.sample();
					if(containsNaNs(store[classNo])) {
						fail("Sampled NaN's");
					}
				}
			}
			else {
				int range = (endIdx-startIdx);
				int startClass1 = startIdx;
				int endClass1 = startIdx + (range / 2);
				int startClass2 = endClass1;
				int endClass2 = endIdx;
				invokeAll(new EJMLSampler(startClass1, endClass1, store, means, cov),
						new EJMLSampler(startClass2, endClass2, store, means, cov));
			}
		}
	}
	
	class EJMLSamplerInv extends RecursiveAction {
		final static long serialVersionUID = 1L;
		int startIdx = -1;
		int endIdx = -1;
		double [][] store;
		int limit = 10;
		double [] means;
		double[][] cov;
		
		public EJMLSamplerInv(int startClass, int endClass, double [][]  store, double [] means, double [][] cov) {
			this.store = store;
			this.startIdx = startClass;
			this.endIdx = endClass;
			this.means = means;
			this.cov = cov;
		}

		@Override
		protected void compute() {
			if ( (endIdx-startIdx) <= limit ) {
				for (int classNo = startIdx; classNo < endIdx; classNo++) {
					FastMVNSamplerEJML sampler = new FastMVNSamplerEJML(means, cov, true);
					store[classNo] = sampler.sample();
					if(containsNaNs(store[classNo])) {
						fail("Sampled NaN's");
					}
				}
			}
			else {
				int range = (endIdx-startIdx);
				int startClass1 = startIdx;
				int endClass1 = startIdx + (range / 2);
				int startClass2 = endClass1;
				int endClass2 = endIdx;
				invokeAll(new EJMLSampler(startClass1, endClass1, store, means, cov),
						new EJMLSampler(startClass2, endClass2, store, means, cov));
			}
		}
	}

	@Test
	public void testParFastMVNEJML() {
		double [][] samples = new double[noLoops][]; 
		long start = System.currentTimeMillis();
		
		double [] means= new double[p]; 
		Arrays.fill(means, 0.0);
		System.out.println("Mean dim: 1 x " + means.length);
		System.out.println("Mean is at: " + means[0] + "x" + means[1]);

		Random rand = new Random();
		DenseMatrix64F covd2 = RandomMatrices.createSymmPosDef(p, rand);
		double [][] cov = MatrixOps.extractDoubleArray(covd2);
		
		EJMLSampler mts = new EJMLSampler(0, noLoops, samples, means, cov);
		
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
	public void testParFastMVNEJMLInv() {
		double [][] samples = new double[noLoops][]; 
		long start = System.currentTimeMillis();
		
		double [] means= new double[p]; 
		Arrays.fill(means, 0.0);
		System.out.println("Mean dim: 1 x " + means.length);
		System.out.println("Mean is at: " + means[0] + "x" + means[1]);

		Random rand = new Random();
		DenseMatrix64F covd2 = RandomMatrices.createSymmPosDef(p, rand);
		org.ejml.ops.CommonOps.invert(covd2);
		double [][] cov = MatrixOps.extractDoubleArray(covd2);
		
		EJMLSamplerInv mts = new EJMLSamplerInv(0, noLoops, samples, means, cov);
		
		ForkJoinPool samplerPool = new ForkJoinPool(Runtime.getRuntime().availableProcessors(), ForkJoinPool.defaultForkJoinWorkerThreadFactory, null, true);
		try {
			samplerPool.invoke(mts);
		} catch (Exception e) {
			e.printStackTrace();
			fail("Exception: " + e);
		}
		System.out.println("Sampling took: " + (System.currentTimeMillis() - start) + " milliseconds...");
	}
	
	class JBLASSampler extends RecursiveAction {
		final static long serialVersionUID = 1L;
		int startIdx = -1;
		int endIdx = -1;
		double [][] store;
		int limit = 10;
		double [] means;
		double[][] cov;
		
		public JBLASSampler(int startClass, int endClass, double [][]  store, double [] means, double [][] cov) {
			this.store = store;
			this.startIdx = startClass;
			this.endIdx = endClass;
			this.means = means;
			this.cov = cov;
		}

		@Override
		protected void compute() {
			if ( (endIdx-startIdx) <= limit ) {
				for (int classNo = startIdx; classNo < endIdx; classNo++) {
					FastMVNSamplerJBLAS sampler = new FastMVNSamplerJBLAS(means, cov);
					store[classNo] = sampler.sample();
					if(containsNaNs(store[classNo])) {
						fail("Sampled NaN's");
					}
				}
			}
			else {
				int range = (endIdx-startIdx);
				int startClass1 = startIdx;
				int endClass1 = startIdx + (range / 2);
				int startClass2 = endClass1;
				int endClass2 = endIdx;
				invokeAll(new JBLASSampler(startClass1, endClass1, store, means, cov),
						new JBLASSampler(startClass2, endClass2, store, means, cov));
			}
		}
	}
	
	class JBLASSamplerInv extends RecursiveAction {
		final static long serialVersionUID = 1L;
		int startIdx = -1;
		int endIdx = -1;
		double [][] store;
		int limit = 10;
		double [] means;
		double[][] cov;
		
		public JBLASSamplerInv(int startClass, int endClass, double [][]  store, double [] means, double [][] cov) {
			this.store = store;
			this.startIdx = startClass;
			this.endIdx = endClass;
			this.means = means;
			this.cov = cov;
		}

		@Override
		protected void compute() {
			if ( (endIdx-startIdx) <= limit ) {
				for (int classNo = startIdx; classNo < endIdx; classNo++) {
					FastMVNSamplerJBLAS sampler = new FastMVNSamplerJBLAS(means, cov, true);
					store[classNo] = sampler.sample();
					if(containsNaNs(store[classNo])) {
						fail("Sampled NaN's");
					}
				}
			}
			else {
				int range = (endIdx-startIdx);
				int startClass1 = startIdx;
				int endClass1 = startIdx + (range / 2);
				int startClass2 = endClass1;
				int endClass2 = endIdx;
				invokeAll(new JBLASSampler(startClass1, endClass1, store, means, cov),
						new JBLASSampler(startClass2, endClass2, store, means, cov));
			}
		}
	}

	private boolean containsNaNs(double[] ds) {
		for (int i = 0; i < ds.length; i++) {
			if(Double.isInfinite(ds[i]) || Double.isNaN(ds[i])) {
				return true; 
			}
		}
		return false;
	}

	@Test
	public void testParFastMVNJBlas() {
		double [][] samples = new double[noLoops][]; 
		long start = System.currentTimeMillis();
		
		double [] means= new double[p]; 
		Arrays.fill(means, 0.0);
		System.out.println("Mean dim: 1 x " + means.length);
		System.out.println("Mean is at: " + means[0] + "x" + means[1]);

		Random rand = new Random();
		DenseMatrix64F covd2 = RandomMatrices.createSymmPosDef(p, rand);
		double [][] cov = MatrixOps.extractDoubleArray(covd2);
		
		JBLASSampler mts = new JBLASSampler(0, noLoops, samples, means, cov);
		
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
	public void testParFastMVNJBlasInv() {
		double [][] samples = new double[noLoops][]; 
		long start = System.currentTimeMillis();
		
		double [] means= new double[p]; 
		Arrays.fill(means, 0.0);
		System.out.println("Mean dim: 1 x " + means.length);
		System.out.println("Mean is at: " + means[0] + "x" + means[1]);

		Random rand = new Random();
		DenseMatrix64F covd2 = RandomMatrices.createSymmPosDef(p, rand);
		org.ejml.ops.CommonOps.invert(covd2);
		double [][] cov = MatrixOps.extractDoubleArray(covd2);
		
		JBLASSampler mts = new JBLASSampler(0, noLoops, samples, means, cov);
		
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
