package xyz.lejon.bayes.models.probit;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

import cc.mallet.topics.LogState;
import cc.mallet.util.LDAUtils;
import xyz.lejon.utils.MatrixOps;

public abstract class AbstractParallelDOSampler extends AbstractDOSampler {
	private static ForkJoinPool pool = new ForkJoinPool();

	/* (non-Javadoc)
	 * @see models.DOSampler#sample(int)
	 */
	@Override
	public void sample(int iterations) {	
		iterationsToRun = iterations;
		for (int iter = 0; iter < iterations; iter++) {
			preIteration(iter);
			currentIteration = iter;
			if(iter % 10 == 0) {
				System.out.println("Iter: " + iter);
				if(printBeta) {
					System.out.println("Betas: " + MatrixOps.doubleArrayToPrintString(betas));
					System.out.println();
				}
			}

			int lowerLimit = 10;
			if(noRows < lowerLimit ) {
				for (int row = 0; row < noRows; row++)
					sampleZ(row);
			} else {
				ZSampler process = new ZSampler(0,noRows,lowerLimit);                
				pool.invoke(process);
			}

			BetaSampler process = new BetaSampler(0,noClasses,1);                
			pool.invoke(process);
			
			if(currentIteration > (((double)burnIn/100)*iterationsToRun)) {
				for (int k = 0; k < noClasses; k++) {
				if(currentIteration % lag  == 0) {
					for (int beta = 0; beta < betas[k].length; beta++) {
						betaMeans[k][beta] += betas[k][beta];
					}
					sampledBetas[k].add(betas[k]);
					noSampledBeta++;
				}
				}
			}

			if(traceBeta) {
				logBetas();
			}
			postIteration(iter);
			if (logLoglikelihood && currentIteration % iterinter == 0) {
				double logLik = doProbitLikelihood();	
				LogState logState = new LogState(logLik, currentIteration, null, loggingPath, null);
				LDAUtils.logLikelihoodToFile(logState);					
			}
		}
		iterationsRun = iterations;
		postSample();
	}

	class ZSampler extends RecursiveAction {
		private static final long serialVersionUID = 1L;
		int startRow = -1;
		int endRow = -1;
		int limit = 1000;

		public ZSampler(int startRow, int endRow, int ll) {
			this.limit = ll;
			this.startRow = startRow;
			this.endRow = endRow;
		}

		public ZSampler(int startRow, int endRow) {
			this.startRow = startRow;
			this.endRow = endRow;
		}

		@Override
		protected void compute() {
			try {
				if ( (endRow-startRow) <= limit ) {
					for (int row = startRow; row < endRow; row++) {
						sampleZ(row);
					}
				}
				else {
					int range = (endRow-startRow);
					int startRow1 = startRow;
					int endRow1 = startRow + (range / 2);
					int startRow2 = endRow1;
					int endRow2 = endRow;
					invokeAll(new ZSampler(startRow1, endRow1, limit),
							new ZSampler(startRow2, endRow2, limit));
				}
			}
			catch ( Exception e ) {
				e.printStackTrace();
			}
		}
	}
	
	class BetaSampler extends RecursiveAction {
		private static final long serialVersionUID = 1L;
		int startClass = -1;
		int endClass = -1;
		int limit = 1000;

		public BetaSampler(int startRow, int endRow, int ll) {
			this.limit = ll;
			this.startClass = startRow;
			this.endClass = endRow;
		}

		public BetaSampler(int startRow, int endRow) {
			this.startClass = startRow;
			this.endClass = endRow;
		}

		@Override
		protected void compute() {
			try {
				if ( (endClass-startClass) <= limit ) {
					for (int j = startClass; j < endClass; j++) {
						sampleBeta(j);
					}
				}
				else {
					int range = (endClass-startClass);
					int startRow1 = startClass;
					int endRow1 = startClass + (range / 2);
					int startRow2 = endRow1;
					int endRow2 = endClass;
					invokeAll(new BetaSampler(startRow1, endRow1, limit),
							new BetaSampler(startRow2, endRow2, limit));
				}
			}
			catch ( Exception e ) {
				e.printStackTrace();
			}
		}
	}
}