package xyz.lejon.bayes.models.dolda;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

import org.apache.commons.configuration.ConfigurationException;

import xyz.lejon.configuration.DOLDAConfiguration;
import xyz.lejon.utils.MatrixOps;

/**
 * THIS CLASS SHOULD CURRENTLY NOT BE USE SINCE JAVA ONLY JBLAS IS NOT THREAD SAFE
 * 
 * This can be used when running with Pure Java BLAS
 * This can be forced by starting Java thusly: 
 * 
 * java -Dcom.github.fommil.netlib.BLAS=com.github.fommil.netlib.F2jBLAS -Dcom.github.fommil.netlib.LAPACK=com.github.fommil.netlib.F2jLAPACK -Dcom.github.fommil.netlib.ARPACK=com.github.fommil.netlib.F2jARPACK
 * 
 * @author Leif Jonsson
 *
 */
public class DOLDAGibbsJBlasHorseshoePar extends DOLDAGibbsJBlasHorseshoe {
	
	private static final long serialVersionUID = 1L;
	private static ForkJoinPool betaSamplerPool;

	public DOLDAGibbsJBlasHorseshoePar(DOLDAConfiguration parentCfg, double[][] xs, int[] ys) throws ConfigurationException {
		super(parentCfg, xs, ys);
		
		// Hopefully everything is synchronized now
		//throw new IllegalStateException("DOLDAGibbsJBlasHorseshoePar SHOULD NOT BE USED SINCE IT TURNS OUT THAT EVEN THE JAVA VERISON OF BLAS USES NON-THREAD SAFE NATIVE CODE!!");
		
		if(!System.getProperty("com.github.fommil.netlib.BLAS").equalsIgnoreCase("com.github.fommil.netlib.F2jBLAS")) {
			throw new IllegalStateException("You are not using the Java version of BLAS, the native is not thread safe and can not be accessed concurrently!!");
		}
		if(!System.getProperty("com.github.fommil.netlib.LAPACK").equalsIgnoreCase("com.github.fommil.netlib.F2jLAPACK")) {
			throw new IllegalStateException("You are not using the Java version of LAPACK, the native is not thread safe and can not be accessed concurrently!!");
		}
		if(!System.getProperty("com.github.fommil.netlib.ARPACK").equalsIgnoreCase("com.github.fommil.netlib.F2jARPACK")) {
			throw new IllegalStateException("You are not using the Java version of ARPACK, the native is not thread safe and can not be accessed concurrently!!");
		}
		betaSamplerPool = new ForkJoinPool();
	}
	
	class BetaSampler extends RecursiveAction {
		final static long serialVersionUID = 1L;
		int startClass = -1;
		int endClass = -1;
		int limit = 1;

		double [][] Xs;
		double [][] XtXs;
		double [][] Xts;
		double [][] precisions;
		

		public BetaSampler(int startClass, int endClass, double [][]  Xs, double [][] XtXs, double [][] Xts, double [][] precisions, int ll) {
			this.Xs = Xs;
			this.XtXs = XtXs;
			this.Xts = Xts;
			this.precisions = precisions;

			this.limit = ll;
			this.startClass = startClass;
			this.endClass = endClass;
			
		}

		@Override
		protected void compute() {
			if ( (endClass-startClass) <= limit ) {
				for (int classNo = startClass; classNo < endClass; classNo++) {
					sampleBeta(Xs, XtXs, Xts, MatrixOps.clone(precisions), classNo);
					saveBetaSample(classNo);
				}
			}
			else {
				int range = (endClass-startClass);
				int startClass1 = startClass;
				int endClass1 = startClass + (range / 2);
				int startClass2 = endClass1;
				int endClass2 = endClass;
				invokeAll(new BetaSampler(startClass1,endClass1, Xs, XtXs, Xts, precisions, limit),
						new BetaSampler(startClass2,endClass2, Xs, XtXs, Xts, precisions, limit));
			}
		}
	}

	@Override
	protected void sampleBetas(double [][] Xs, double [][] XtXs, double [][] Xts, double [][] precisions) {
		BetaSampler bs = new BetaSampler(0, noClasses,Xs, XtXs, Xts, precisions,10);                
		betaSamplerPool.invoke(bs); 
	}

}
