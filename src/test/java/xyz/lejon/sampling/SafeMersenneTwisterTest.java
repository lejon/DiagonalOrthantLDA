package xyz.lejon.sampling;

import static org.junit.Assert.fail;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

import org.junit.Test;
import org.junit.experimental.categories.Category;

import xyz.lejon.MarkerIFParallelTest;

@Category(MarkerIFParallelTest.class)
public class SafeMersenneTwisterTest {
	
	static jdistlib.rng.RandomEngine sharedMt = new SafeMersenneTwister();	
	
	class MTSampler extends RecursiveAction {
		final static long serialVersionUID = 1L;
		int startIdx = -1;
		int endIdx = -1;
		double [] store;
		int limit = 10;
		
		public MTSampler(int startClass, int endClass, double []  store) {
			this.store = store;
			this.startIdx = startClass;
			this.endIdx = endClass;
		}

		@Override
		protected void compute() {
			if ( (endIdx-startIdx) <= limit ) {
				for (int classNo = startIdx; classNo < endIdx; classNo++) {
					for (int i = 0; i < 1000; i++) {							
						store[classNo] = jdistlib.Uniform.random(0, 60, sharedMt);
					}
				}
			}
			else {
				int range = (endIdx-startIdx);
				int startClass1 = startIdx;
				int endClass1 = startIdx + (range / 2);
				int startClass2 = endClass1;
				int endClass2 = endIdx;
				invokeAll(new MTSampler(startClass1, endClass1, store),
						new MTSampler(startClass2, endClass2, store));
			}
		}
	}

	@Test
	public void testParJDistlibUnif1() {
		//int noLoops = 150_000_000;
		//int noLoops = 15_000_000;
		int noLoops = 300_000;
		double [] samples = new double[noLoops]; 
		long start = System.currentTimeMillis();
		MTSampler mts = new MTSampler(0, noLoops, samples);
		
		ForkJoinPool samplerPool = new ForkJoinPool();
		try {
			samplerPool.invoke(mts);
		} catch (Exception e) {
			e.printStackTrace();
			fail("Exception: " + e);
		}
		System.out.println("Sampling took: " + (System.currentTimeMillis() - start) + " milliseconds...");
	}
}
