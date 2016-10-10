package xyz.lejon.runnables;

// To run with Pure Java implementation of BLAS, LAPACK and ARPACK use the below command line
// If this is done, you can use the DOLDAGibbsJBlasHorseshoePar to parallelize the Beta sampling (per class)
// On the mozilla-25000 dataset, this reduced the sampling of 10 iterations from 7 to 4 minutes!
// java -Xmx8g -Dcom.github.fommil.netlib.BLAS=com.github.fommil.netlib.F2jBLAS -Dcom.github.fommil.netlib.LAPACK=com.github.fommil.netlib.F2jLAPACK -Dcom.github.fommil.netlib.ARPACK=com.github.fommil.netlib.F2jARPACK -cp DOLDA-0.6.2.jar xyz.lejon.runnables.DOLDAClassification --normalize --run_cfg=DOLDAConfigs/DOLDAClassification.cfg

import xyz.lejon.bayes.models.dolda.DOLDAClassifier;
import xyz.lejon.bayes.models.dolda.DOLDADataSet;
import xyz.lejon.bayes.models.dolda.DOLDASamplingClassifier;
import xyz.lejon.configuration.DOLDAConfiguration;

/** 
 * A hack class for running a completing fold if the last fold crashed
 * 
 * @author Leif Jonsson
 *
 */
public class CompleteDOLDAClassificationDistribution extends CompleteDOLDAClassification {

	public static void main(String[] args) throws Exception {
		CompleteDOLDAClassificationDistribution dc = new CompleteDOLDAClassificationDistribution();
		dc.execute(args);
	}
	
	@Override
	protected DOLDAClassifier getSamplingClassifier(DOLDAConfiguration config,
			DOLDADataSet trainingSetData) {
		return new DOLDASamplingClassifier(config, trainingSetData);
	}
}
