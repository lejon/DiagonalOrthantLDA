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
public class MVNSamplerSharedTest {
	
	int noLoops = 1000;
	int p = 1000;
		
	class EJMLSampler extends RecursiveAction {
		final static long serialVersionUID = 1L;
		int startIdx = -1;
		int endIdx = -1;
		double [][] store;
		int limit = 10;
		FastMVNSamplerEJML sampler;
		
		public EJMLSampler(int startClass, int endClass, double [][]  store, FastMVNSamplerEJML sampler) {
			this.store = store;
			this.startIdx = startClass;
			this.endIdx = endClass;
			this.sampler = sampler;
		}

		@Override
		protected void compute() {
			if ( (endIdx-startIdx) <= limit ) {
				//System.out.println("Sampling between " + startIdx + " and " + endIdx);
				for (int classNo = startIdx; classNo < endIdx; classNo++) {
					store[classNo] = sampler.sample();
					//System.out.println(Thread.currentThread().getId() + ": Sampled: " + MatrixOps.arrToStr(store[classNo]));
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
				invokeAll(new EJMLSampler(startClass1, endClass1, store, sampler),
						new EJMLSampler(startClass2, endClass2, store, sampler));
			}
		}
	}
	
	class EJMLSamplerInv extends RecursiveAction {
		final static long serialVersionUID = 1L;
		int startIdx = -1;
		int endIdx = -1;
		double [][] store;
		int limit = 10;
		FastMVNSamplerEJML sampler;
		
		public EJMLSamplerInv(int startClass, int endClass, double [][]  store, FastMVNSamplerEJML sampler) {
			this.store = store;
			this.startIdx = startClass;
			this.endIdx = endClass;
			this.sampler = sampler;
		}

		@Override
		protected void compute() {
			if ( (endIdx-startIdx) <= limit ) {
				//System.out.println(Thread.currentThread().getId() + ": Sampling between " + startIdx + " and " + endIdx);
				for (int classNo = startIdx; classNo < endIdx; classNo++) {
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
				invokeAll(new EJMLSampler(startClass1, endClass1, store, sampler),
						new EJMLSampler(startClass2, endClass2, store, sampler));
			}
		}
	}

	@Test
	public void testParFastMVNEJML() {
		double [][] samples = new double[noLoops][]; 
		//long start = System.currentTimeMillis();
		
		double [] means= new double[p]; 
		Arrays.fill(means, 0.0);
		//System.out.println("Mean dim: 1 x " + means.length);
		//System.out.println("Mean is at: " + means[0] + "x" + means[1]);

		Random rand = new Random();
		DenseMatrix64F covd2 = RandomMatrices.createSymmPosDef(p, rand);
		double [][] cov = MatrixOps.extractDoubleArray(covd2);
		
		FastMVNSamplerEJML sampler = new FastMVNSamplerEJML(means, cov);
		EJMLSampler mts = new EJMLSampler(0, noLoops, samples, sampler);
		
		ForkJoinPool samplerPool = new ForkJoinPool(Runtime.getRuntime().availableProcessors(), ForkJoinPool.defaultForkJoinWorkerThreadFactory, null, true);
		try {
			samplerPool.invoke(mts);
		} catch (Exception e) {
			e.printStackTrace();
			fail("Exception: " + e);
		}
		//System.out.println("Sampling took: " + (System.currentTimeMillis() - start) + " milliseconds...");
	}
	
	@Test
	public void testParFastMVNEJMLInv() {
		double [][] samples = new double[noLoops][]; 
		//long start = System.currentTimeMillis();
		
		double [] means= new double[p]; 
		Arrays.fill(means, 0.0);
		//System.out.println("Mean dim: 1 x " + means.length);
		//System.out.println("Mean is at: " + means[0] + "x" + means[1]);

		Random rand = new Random();
		DenseMatrix64F covd2 = RandomMatrices.createSymmPosDef(p, rand);
		org.ejml.ops.CommonOps.invert(covd2);
		double [][] cov = MatrixOps.extractDoubleArray(covd2);
		
		FastMVNSamplerEJML sampler = new FastMVNSamplerEJML(means, cov, true);
		
		EJMLSamplerInv mts = new EJMLSamplerInv(0, noLoops, samples, sampler);
		
		ForkJoinPool samplerPool = new ForkJoinPool(Runtime.getRuntime().availableProcessors(), ForkJoinPool.defaultForkJoinWorkerThreadFactory, null, true);
		try {
			samplerPool.invoke(mts);
		} catch (Exception e) {
			e.printStackTrace();
			fail("Exception: " + e);
		}
		//System.out.println("Sampling took: " + (System.currentTimeMillis() - start) + " milliseconds...");
	}
	
	class JBLASSampler extends RecursiveAction {
		final static long serialVersionUID = 1L;
		int startIdx = -1;
		int endIdx = -1;
		double [][] store;
		int limit = 10;
		FastMVNSamplerJBLAS sampler;
		
		public JBLASSampler(int startClass, int endClass, double [][]  store, FastMVNSamplerJBLAS sampler) {
			this.store = store;
			this.startIdx = startClass;
			this.endIdx = endClass;
			this.sampler = sampler;
		}

		@Override
		protected void compute() {
			if ( (endIdx-startIdx) <= limit ) {
				//System.out.println(Thread.currentThread().getId() + ": Sampling between " + startIdx + " and " + endIdx);
				for (int classNo = startIdx; classNo < endIdx; classNo++) {
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
				invokeAll(new JBLASSampler(startClass1, endClass1, store, sampler),
						new JBLASSampler(startClass2, endClass2, store, sampler));
			}
		}
	}
	
	class JBLASSamplerInv extends RecursiveAction {
		final static long serialVersionUID = 1L;
		int startIdx = -1;
		int endIdx = -1;
		double [][] store;
		int limit = 10;
		FastMVNSamplerJBLAS sampler;
		
		public JBLASSamplerInv(int startClass, int endClass, double [][]  store, FastMVNSamplerJBLAS sampler) {
			this.store = store;
			this.startIdx = startClass;
			this.endIdx = endClass;
			this.sampler = sampler;
		}


		@Override
		protected void compute() {
			if ( (endIdx-startIdx) <= limit ) {
				//System.out.println(Thread.currentThread().getId() + ": Sampling between " + startIdx + " and " + endIdx);
				for (int classNo = startIdx; classNo < endIdx; classNo++) {
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
				invokeAll(new JBLASSampler(startClass1, endClass1, store, sampler),
						new JBLASSampler(startClass2, endClass2, store, sampler));
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
		//long start = System.currentTimeMillis();
		
		double [] means= new double[p]; 
		Arrays.fill(means, 0.0);
		//System.out.println("Mean dim: 1 x " + means.length);
		//System.out.println("Mean is at: " + means[0] + "x" + means[1]);

		Random rand = new Random();
		DenseMatrix64F covd2 = RandomMatrices.createSymmPosDef(p, rand);
		double [][] cov = MatrixOps.extractDoubleArray(covd2);
		
		FastMVNSamplerJBLAS sampler = new FastMVNSamplerJBLAS(means, cov);
		
		JBLASSampler mts = new JBLASSampler(0, noLoops, samples, sampler);
		
		ForkJoinPool samplerPool = new ForkJoinPool(Runtime.getRuntime().availableProcessors(), ForkJoinPool.defaultForkJoinWorkerThreadFactory, null, true);
		try {
			samplerPool.invoke(mts);
		} catch (Exception e) {
			e.printStackTrace();
			fail("Exception: " + e);
		}
		//System.out.println("Sampling took: " + (System.currentTimeMillis() - start) + " milliseconds...");
	}
	
	@Test
	public void testParFastMVNJBlasInv() {
		double [][] samples = new double[noLoops][]; 
		//long start = System.currentTimeMillis();
		
		double [] means= new double[p]; 
		Arrays.fill(means, 0.0);
		//System.out.println("Mean dim: 1 x " + means.length);
		//System.out.println("Mean is at: " + means[0] + "x" + means[1]);

		Random rand = new Random();
		DenseMatrix64F covd2 = RandomMatrices.createSymmPosDef(p, rand);
		org.ejml.ops.CommonOps.invert(covd2);
		double [][] cov = MatrixOps.extractDoubleArray(covd2);
		
		FastMVNSamplerJBLAS sampler = new FastMVNSamplerJBLAS(means, cov, true);
		
		JBLASSampler mts = new JBLASSampler(0, noLoops, samples, sampler);
		
		ForkJoinPool samplerPool = new ForkJoinPool(Runtime.getRuntime().availableProcessors(), ForkJoinPool.defaultForkJoinWorkerThreadFactory, null, true);
		try {
			samplerPool.invoke(mts);
		} catch (Exception e) {
			e.printStackTrace();
			fail("Exception: " + e);
		}
		//System.out.println("Sampling took: " + (System.currentTimeMillis() - start) + " milliseconds...");
	}
}
