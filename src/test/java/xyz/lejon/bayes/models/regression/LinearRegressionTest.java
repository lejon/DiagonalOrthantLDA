package xyz.lejon.bayes.models.regression;

import static org.junit.Assert.assertTrue;

import java.io.IOException;

import org.apache.commons.cli.ParseException;
import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import xyz.lejon.MarkerIFSmokeTest;
import xyz.lejon.configuration.DOConfiguration;
import xyz.lejon.configuration.OLSCommandLineParser;
import xyz.lejon.configuration.OLSConfigFactory;
import xyz.lejon.configuration.OLSConfiguration;
import xyz.lejon.utils.BlasOps;
import xyz.lejon.utils.LoggingUtils;
import xyz.lejon.utils.MatrixOps;

@Category(MarkerIFSmokeTest.class)
public class LinearRegressionTest {

	KolmogorovSmirnovTest ks = new KolmogorovSmirnovTest();

	@Test
	public void testLinearRegressionHS() throws IOException, ParseException, ConfigurationException {
		String [] args = {"--run_cfg=src/main/resources/configuration/LinearRegressionUnitTest.cfg"};
		OLSCommandLineParser cp = new OLSCommandLineParser(args);
		OLSConfiguration config = (OLSConfiguration) OLSConfigFactory.getMainConfiguration(cp);
		LoggingUtils lu = new LoggingUtils();
		config.setLoggingUtil(lu);

		String conf = "OLS_Diabetes_HS";

		lu.checkCreateAndSetSubLogDir(conf);
		config.activateSubconfig(conf);

		System.out.println("Running SubConf: " + conf);
		System.out.println("Using Config: " + config.whereAmI());
		String dataset_fn = config.getDatasetFilename();
		System.out.println("Using dataset: " + dataset_fn);
		config.loadTrainingSet();
		config.loadTestSet();
		System.out.println("Using lag: " + config.getLag());
		System.out.println("Using burnIn: " + config.getBurnIn());

		double [][] xs = config.getX();
		double [] ys = config.getY();		

		LinearRegression linearRegression = ModelFactory.get(config, xs, ys);

		System.out.println("Using sampler: " + linearRegression.getClass().getName());
		System.out.println("X is: " + MatrixOps.doubleArrayToPrintString(xs, 5));
		System.out.println("y is: " + MatrixOps.arrToStr(ys, 10));

		linearRegression.sample(config.getNoIterations(DOConfiguration.ITERATIONS_DEFAULT));

		double [] betas = linearRegression.getBetas();


		double [] stan_hs = { 0.0032995587, -0.0426264899,  0.3540063989,  0.1362060672, -0.0312783076, -0.0066871019, -0.0854391403,  0.0256493184,  0.3049474571,
				0.0142994631,  0.0097661554,  0.0220807705,  0.0125896495, -0.0090949496, -0.0084324993,  0.0005084726,  0.0160970762, -0.0163315695,
				0.0153123700,  0.0666900170,  0.0032630260,  0.0178220782, -0.0108003922, -0.0262887385,  0.0029952220,  0.0054490074,  0.0438141137,
				0.0161605573,  0.0184329370,  0.0192614566,  0.0045031642, -0.0013621072,  0.0088406142,  0.0002122676,  0.0016053109,  0.0109659420,
				0.0202642093, -0.0049132633, -0.0016929414,  0.0016653807,  0.0012817871,  0.0007484448,  0.0207992422,  0.0101263736,  0.0093455587,
				0.0038156285,  0.0033428363,  0.0147496179, -0.0031897211, -0.0014353205,  0.0052557137, -0.0092481666, -0.0158275559,  0.0097967785,
				-0.0057631777,  0.0075607376,  0.0067311591,  0.0105193574, -0.0051074657,  0.0166227056,  0.0057195071, -0.0248496015,  0.0068876185,
				0.0030807274};


		KolmogorovSmirnovTest ks = new KolmogorovSmirnovTest();
		boolean ksResult = ks.kolmogorovSmirnovTest(betas, stan_hs) > 0.00001;
		if(ksResult) {
			System.out.println("OK!");	
		} else {
			System.out.println("NOK!");
		}
		System.out.println("Draws:");
		for (int j = 0; j < stan_hs.length; j++) {
			System.out.print(stan_hs[j]  + "<=>" + betas[j] + ", ");
		}
		System.out.println();
		assertTrue(ksResult);
	}

	@Test
	public void testLinearRegressionNormal() throws IOException, ParseException, ConfigurationException {
		String [] args = {"--run_cfg=src/main/resources/configuration/LinearRegressionUnitTest.cfg"};
		OLSCommandLineParser cp = new OLSCommandLineParser(args);
		OLSConfiguration config = (OLSConfiguration) OLSConfigFactory.getMainConfiguration(cp);
		LoggingUtils lu = new LoggingUtils();
		config.setLoggingUtil(lu);

		String conf = "OLS_Diabetes_Normal";

		lu.checkCreateAndSetSubLogDir(conf);
		config.activateSubconfig(conf);

		System.out.println("Running SubConf: " + conf);
		System.out.println("Using Config: " + config.whereAmI());
		String dataset_fn = config.getDatasetFilename();
		System.out.println("Using dataset: " + dataset_fn);
		config.loadTrainingSet();
		config.loadTestSet();
		System.out.println("Using lag: " + config.getLag());
		System.out.println("Using burnIn: " + config.getBurnIn());

		double [][] xs = config.getX();
		double [] ys = config.getY();		

		LinearRegression linearRegression = ModelFactory.get(config, xs, ys);

		System.out.println("Using sampler: " + linearRegression.getClass().getName());
		System.out.println("X is: " + MatrixOps.doubleArrayToPrintString(xs, 5));
		System.out.println("y is: " + MatrixOps.arrToStr(ys, 10));

		linearRegression.sample(config.getNoIterations(DOConfiguration.ITERATIONS_DEFAULT));

		double [] betas = linearRegression.getBetas();


		double [] r_ols = { -0.009659686,  0.034510192, -0.119843030,  0.281272003, 0.158404023, -7.709359668, 6.727818322,
				2.761911601,  0.028932318,  2.909802110,  0.073729411,  0.024773627,  0.048165853,  0.030244618, 5.555899258,  
				2.333582765,  1.145512276,  0.556219176,  1.244427518,  0.034223667,  0.085211252,	-0.028246293,  0.042587370, 
				-0.209505066, -0.024965752,  0.191285504,  0.160688008,  0.176622141, 0.026875756,  0.054896180,  0.055449950,  
				0.542381922, -0.418896210, -0.172196073, -0.073718105, -0.174683606,  0.013659800,  0.055915202, -0.468269051,
				0.398116522, 0.171149593, -0.064612652, 0.163538010,  0.069540412,  0.365978724, -0.205956113, -0.168052231,
				-0.077806217, -0.064715735,	-0.141659547, -7.338368932, -2.882222273, -2.049815787, -2.204253573,  0.330447880,  
				1.878185343, 1.664927506,  1.261846069, -0.242166584,  1.017243679,  0.829593048, -0.101326692,  0.489019741, 
				0.003657258, -0.047164160};


		KolmogorovSmirnovTest ks = new KolmogorovSmirnovTest();
		boolean ksResult = ks.kolmogorovSmirnovTest(betas, r_ols) > 0.00001;
		if(ksResult) {
			System.out.println("OK!");	
		} else {
			System.out.println("NOK!");
		}
		System.out.println("Draws:");
		for (int j = 0; j < r_ols.length; j++) {
			System.out.println(r_ols[j]  + "<=>" + betas[j] + ", ");
		}
		System.out.println();
		assertTrue(ksResult);
	}

	@Test
	public void testOLS() throws IOException, ParseException, ConfigurationException {
		String [] args = {"--run_cfg=src/main/resources/configuration/LinearRegressionUnitTest.cfg"};
		OLSCommandLineParser cp = new OLSCommandLineParser(args);
		OLSConfiguration config = (OLSConfiguration) OLSConfigFactory.getMainConfiguration(cp);
		LoggingUtils lu = new LoggingUtils();
		config.setLoggingUtil(lu);

		String conf = "OLS_Diabetes_HS";

		lu.checkCreateAndSetSubLogDir(conf);
		config.activateSubconfig(conf);

		System.out.println("Running SubConf: " + conf);
		System.out.println("Using Config: " + config.whereAmI());
		String dataset_fn = config.getDatasetFilename();
		System.out.println("Using dataset: " + dataset_fn);
		config.loadTrainingSet();
		config.loadTestSet();
		System.out.println("Using lag: " + config.getLag());
		System.out.println("Using burnIn: " + config.getBurnIn());

		double [][] xs = config.getX();
		double [] ys = config.getY();		

		DoubleMatrix Xd = new DoubleMatrix(xs);
		DoubleMatrix Yd = new DoubleMatrix(ys);
		DoubleMatrix Xdt = (new DoubleMatrix(xs)).transpose();
		DoubleMatrix XtX = Xdt.mmul(Xd);

		
		// OLS estimate
		DoubleMatrix betahat = BlasOps.blasInvert(XtX).mmul(Xdt).mmul(Yd);
		double [] betas = betahat.toArray();
		System.out.println("BetaHat = " + MatrixOps.arrToStr(betas));
		
		double [] rlm = {0.03450307, -0.11994839,  0.28162376,  0.15798554, -7.92325426,  6.91719643,  2.84079377,  0.02558989,  2.98105144, 
				0.07360629,  0.02401240,  0.04833213,  0.03037863,  5.52469013,  2.32648356,  1.13330107,  0.55958626,  1.25183982, 
				0.03458122,  0.08522647, -0.02804902,  0.04307128, -0.21561792, -0.01968803,  0.19378929,  0.16147329,  0.17828597, 
				0.02682765,  0.05478299,  0.05623759,  0.54407872, -0.42090195, -0.17300980, -0.07340830, -0.17470747,  0.01301164, 
				0.05600984, -0.46355952,  0.39415956,  0.16892666, -0.06566154,  0.16153389,  0.06998010,  0.37578270, -0.21391034, 
				-0.17156237, -0.07816966, -0.06824226, -0.14186633, -7.30325992, -2.85664442, -2.03051413, -2.15610028,  0.32507753, 
				1.86068016,  1.64633358,  1.22688387, -0.23816942,  1.01147172,  0.80864178, -0.09900410,  0.48073475,  0.00570288, 
				-0.04639973};
		

		KolmogorovSmirnovTest ks = new KolmogorovSmirnovTest();
		boolean ksResult = ks.kolmogorovSmirnovTest(betas, rlm) > 0.00001;
		if(ksResult) {
			System.out.println("OK!");	
		} else {
			System.out.println("NOK!");
		}
		System.out.println("Draws:");
		for (int j = 0; j < rlm.length; j++) {
			System.out.print(rlm[j]  + "<=>" + betas[j] + ", ");
		}
		System.out.println();
		assertTrue(ksResult);
	}
	
	@Test
	public void testOLSSim() throws IOException, ParseException, ConfigurationException {
		String [] args = {"--run_cfg=src/main/resources/configuration/LinearRegressionUnitTest.cfg"};
		OLSCommandLineParser cp = new OLSCommandLineParser(args);
		OLSConfiguration config = (OLSConfiguration) OLSConfigFactory.getMainConfiguration(cp);
		LoggingUtils lu = new LoggingUtils();
		config.setLoggingUtil(lu);

		String conf = "OLS_Simulated";

		lu.checkCreateAndSetSubLogDir(conf);
		config.activateSubconfig(conf);

		System.out.println("Running SubConf: " + conf);
		System.out.println("Using Config: " + config.whereAmI());
		String dataset_fn = config.getDatasetFilename();
		System.out.println("Using dataset: " + dataset_fn);
		config.loadTrainingSet();
		config.loadTestSet();
		System.out.println("Using lag: " + config.getLag());
		System.out.println("Using burnIn: " + config.getBurnIn());

		double [][] xs = config.getX();
		double [] ys = config.getY();		

		DoubleMatrix Xd = new DoubleMatrix(xs);
		DoubleMatrix Yd = new DoubleMatrix(ys);
		DoubleMatrix Xdt = (new DoubleMatrix(xs)).transpose();
		DoubleMatrix XtX = Xdt.mmul(Xd);

		
		// OLS estimate
		DoubleMatrix betahat = BlasOps.blasInvert(XtX).mmul(Xdt).mmul(Yd);
		double [] betas = betahat.toArray();
		System.out.println("BetaHat = " + MatrixOps.arrToStr(betas));
		
		// b0 <- 17, b1 <- 0.5, b2 <- 0.037, b3 <- -5.2, sigma <- 1.4
		double [] betaTrue = {17,0.5,0.037,-5.2};
		

		KolmogorovSmirnovTest ks = new KolmogorovSmirnovTest();
		boolean ksResult = ks.kolmogorovSmirnovTest(betas, betaTrue) > 0.00001;
		if(ksResult) {
			System.out.println("OK!");	
		} else {
			System.out.println("NOK!");
		}
		System.out.println("Draws:");
		for (int j = 0; j < betaTrue.length; j++) {
			System.out.print(betaTrue[j]  + "<=>" + betas[j] + ", ");
		}
		System.out.println();
		assertTrue(ksResult);
	}
	
	@Test
	public void testLinearRegressionNormalSimulated() throws IOException, ParseException, ConfigurationException {
		String [] args = {"--run_cfg=src/main/resources/configuration/LinearRegressionUnitTest.cfg"};
		OLSCommandLineParser cp = new OLSCommandLineParser(args);
		OLSConfiguration config = (OLSConfiguration) OLSConfigFactory.getMainConfiguration(cp);
		LoggingUtils lu = new LoggingUtils();
		config.setLoggingUtil(lu);

		String conf = "OLS_Simulated";

		lu.checkCreateAndSetSubLogDir(conf);
		config.activateSubconfig(conf);

		System.out.println("Running SubConf: " + conf);
		System.out.println("Using Config: " + config.whereAmI());
		String dataset_fn = config.getDatasetFilename();
		System.out.println("Using dataset: " + dataset_fn);
		config.loadTrainingSet();
		config.loadTestSet();
		System.out.println("Using lag: " + config.getLag());
		System.out.println("Using burnIn: " + config.getBurnIn());

		double [][] xs = config.getX();
		double [] ys = config.getY();		

		LinearRegression linearRegression = ModelFactory.get(config, xs, ys);

		System.out.println("Using sampler: " + linearRegression.getClass().getName());
		System.out.println("X is: " + MatrixOps.doubleArrayToPrintString(xs, 5));
		System.out.println("y is: " + MatrixOps.arrToStr(ys, 10));

		linearRegression.sample(config.getNoIterations(DOConfiguration.ITERATIONS_DEFAULT));

		double [] betas = linearRegression.getBetas();


		// b0 <- 17, b1 <- 0.5, b2 <- 0.037, b3 <- -5.2, sigma <- 1.4
		double [] betaTrue = {17,0.5,0.037,-5.2};


		KolmogorovSmirnovTest ks = new KolmogorovSmirnovTest();
		boolean ksResult = ks.kolmogorovSmirnovTest(betas, betaTrue) > 0.00001;
		if(ksResult) {
			System.out.println("OK!");	
		} else {
			System.out.println("NOK!");
		}
		System.out.println("Draws:");
		for (int j = 0; j < betaTrue.length; j++) {
			System.out.println(betaTrue[j]  + "<=>" + betas[j] + ", ");
		}
		System.out.println();
		assertTrue(ksResult);
	}
		
	@Test
	public void testLinearRegressionHSSimulatedEJML() throws IOException, ParseException, ConfigurationException {
		String [] args = {"--run_cfg=src/main/resources/configuration/LinearRegressionUnitTest.cfg"};
		OLSCommandLineParser cp = new OLSCommandLineParser(args);
		OLSConfiguration config = (OLSConfiguration) OLSConfigFactory.getMainConfiguration(cp);
		LoggingUtils lu = new LoggingUtils();
		config.setLoggingUtil(lu);

		String conf = "OLS_Simulated";

		lu.checkCreateAndSetSubLogDir(conf);
		config.activateSubconfig(conf);

		System.out.println("Running SubConf: " + conf);
		System.out.println("Using Config: " + config.whereAmI());
		String dataset_fn = config.getDatasetFilename();
		System.out.println("Using dataset: " + dataset_fn);
		config.loadTrainingSet();
		config.loadTestSet();
		System.out.println("Using lag: " + config.getLag());
		System.out.println("Using burnIn: " + config.getBurnIn());

		double [][] xs = config.getX();
		double [] ys = config.getY();		

		LinearRegression linearRegression = new LinearRegressionEJMLHSPrior(config, xs, ys);

		System.out.println("Using sampler: " + linearRegression.getClass().getName());
		System.out.println("X is: " + MatrixOps.doubleArrayToPrintString(xs, 5));
		System.out.println("y is: " + MatrixOps.arrToStr(ys, 10));

		linearRegression.sample(config.getNoIterations(DOConfiguration.ITERATIONS_DEFAULT));

		double [] betas = linearRegression.getBetas();


		// b0 <- 17, b1 <- 0.5, b2 <- 0.037, b3 <- -5.2, sigma <- 1.4
		double [] betaTrue = {17,0.5,0.037,-5.2};


		KolmogorovSmirnovTest ks = new KolmogorovSmirnovTest();
		boolean ksResult = ks.kolmogorovSmirnovTest(betas, betaTrue) > 0.00001;
		if(ksResult) {
			System.out.println("OK!");	
		} else {
			System.out.println("NOK!");
		}
		System.out.println("Draws:");
		for (int j = 0; j < betaTrue.length; j++) {
			System.out.print(betaTrue[j]  + "<=>" + betas[j] + ", ");
		}
		System.out.println();
		assertTrue(ksResult);
	}
	
	@Test
	public void testLinearRegressionHSSimulatedBLAS() throws IOException, ParseException, ConfigurationException {
		String [] args = {"--run_cfg=src/main/resources/configuration/LinearRegressionUnitTest.cfg"};
		OLSCommandLineParser cp = new OLSCommandLineParser(args);
		OLSConfiguration config = (OLSConfiguration) OLSConfigFactory.getMainConfiguration(cp);
		LoggingUtils lu = new LoggingUtils();
		config.setLoggingUtil(lu);

		String conf = "OLS_Simulated";

		lu.checkCreateAndSetSubLogDir(conf);
		config.activateSubconfig(conf);

		System.out.println("Running SubConf: " + conf);
		System.out.println("Using Config: " + config.whereAmI());
		String dataset_fn = config.getDatasetFilename();
		System.out.println("Using dataset: " + dataset_fn);
		config.loadTrainingSet();
		config.loadTestSet();
		System.out.println("Using lag: " + config.getLag());
		System.out.println("Using burnIn: " + config.getBurnIn());

		double [][] xs = config.getX();
		double [] ys = config.getY();		

		LinearRegression linearRegression = new LinearRegressionJBlasHSPrior(config, xs, ys);

		System.out.println("Using sampler: " + linearRegression.getClass().getName());
		System.out.println("X is: " + MatrixOps.doubleArrayToPrintString(xs, 5));
		System.out.println("y is: " + MatrixOps.arrToStr(ys, 10));

		linearRegression.sample(config.getNoIterations(DOConfiguration.ITERATIONS_DEFAULT));

		double [] betas = linearRegression.getBetas();


		// b0 <- 17, b1 <- 0.5, b2 <- 0.037, b3 <- -5.2, sigma <- 1.4
		double [] betaTrue = {17,0.5,0.037,-5.2};


		KolmogorovSmirnovTest ks = new KolmogorovSmirnovTest();
		boolean ksResult = ks.kolmogorovSmirnovTest(betas, betaTrue) > 0.00001;
		if(ksResult) {
			System.out.println("OK!");	
		} else {
			System.out.println("NOK!");
		}
		System.out.println("Draws:");
		for (int j = 0; j < betaTrue.length; j++) {
			System.out.print(betaTrue[j]  + "<=>" + betas[j] + ", ");
		}
		System.out.println();
		assertTrue(ksResult);
	}

	@Test
	public void testBetas() {
		double [] monomvn = {0.0021901279, -0.0362873005,  0.3593992509,  0.1355886462, -0.0323761412, -0.0055987091, -0.0776761781,  0.0215820988,  0.3064267605,  0.0150554738, 
				0.0080563397,  0.0188819247,  0.0106878883, -0.0047959825, -0.0072562627, -0.0019842287,  0.0157489017, -0.0155504750,  0.0116158728,  0.0625663600, 
				0.0018334167,  0.0172995111, -0.0082754491, -0.0252153519,  0.0015649068,  0.0046757300,  0.0387122694,  0.0188604076,  0.0180570099,  0.0186286038, 
				0.0057644593, -0.0021003258,  0.0076035304,  0.0003643636,  0.0023280750,  0.0109034593,  0.0182495209, -0.0045486809, -0.0022382566,  0.0011825617, 
				0.0010624732,  0.0026851888,  0.0195123848,  0.0089726110,  0.0078981753,  0.0035199391,  0.0039503554,  0.0145961381, -0.0016826040, -0.0047452202, 
				0.0067934547, -0.0111693342, -0.0132251509,  0.0104678054, -0.0051948855,  0.0088524009,  0.0032063088,  0.0094083619, -0.0041539877,  0.0178477091, 
				0.0043937267, -0.0245193128,  0.0055946208,  0.0027125329};

		double [] stan_hs = { 0.0032995587, -0.0426264899,  0.3540063989,  0.1362060672, -0.0312783076, -0.0066871019, -0.0854391403,  0.0256493184,  0.3049474571,
				0.0142994631,  0.0097661554,  0.0220807705,  0.0125896495, -0.0090949496, -0.0084324993,  0.0005084726,  0.0160970762, -0.0163315695,
				0.0153123700,  0.0666900170,  0.0032630260,  0.0178220782, -0.0108003922, -0.0262887385,  0.0029952220,  0.0054490074,  0.0438141137,
				0.0161605573,  0.0184329370,  0.0192614566,  0.0045031642, -0.0013621072,  0.0088406142,  0.0002122676,  0.0016053109,  0.0109659420,
				0.0202642093, -0.0049132633, -0.0016929414,  0.0016653807,  0.0012817871,  0.0007484448,  0.0207992422,  0.0101263736,  0.0093455587,
				0.0038156285,  0.0033428363,  0.0147496179, -0.0031897211, -0.0014353205,  0.0052557137, -0.0092481666, -0.0158275559,  0.0097967785,
				-0.0057631777,  0.0075607376,  0.0067311591,  0.0105193574, -0.0051074657,  0.0166227056,  0.0057195071, -0.0248496015,  0.0068876185,
				0.0030807274};

		double [] mons = {0.0015340941, -0.0400614055,  0.3556591164,  0.1402965766, -0.0266504007, -0.0104125522, -0.0828667233,  0.0232541457,  0.3023416386,
				0.0145399398,  0.0098541583,  0.0210143797,  0.0143499126, -0.0065452417, -0.0087228537, -0.0012028997,  0.0160917728, -0.0143621430,
				0.0108168867,  0.0639845294,  0.0036456970,  0.0199240433, -0.0086280941, -0.0297119478,  0.0014586141,  0.0065346124,  0.0381407804,
				0.0130343700,  0.0185575074,  0.0168804396,  0.0055292620, -0.0025521755,  0.0086815627,  0.0002659110,  0.0009008475,  0.0126272009,
				0.0204262685, -0.0048089241, -0.0007516699,  0.0018296036,  0.0004044271,  0.0024754171,  0.0188534755,  0.0094405017,  0.0085972458,
				0.0026204096,  0.0024166633,  0.0104008469, -0.0016545261, -0.0035586122,  0.0061012859, -0.0053475208, -0.0132026210,  0.0083632133,
				-0.0053187688,  0.0058895604,  0.0051466727,  0.0096159101, -0.0062332487,  0.0151781304,  0.0040664832, -0.0261134130,  0.0065639495,
				0.0041348292}; 

		double [] java1 = {0.0024867575, -0.0298174560,  0.3791783713, 0.1201065370, -0.0226223413, -0.0028209850, -0.0659370091,  0.0200771945,  0.3075865232, 
				0.0186519957,  0.0115806732,  0.0145161835, 0.0075595308, -0.0049207513, -0.0074744381, -0.0009715514,  0.0141518982, -0.0107275622, 
				0.0155370914,  0.0618954381,  0.0026627645, 0.0140148535, -0.0007934979, -0.0192020053,  0.0006718793,  0.0086247579,  0.0265034445, 
				0.0144066838,  0.0132861822,  0.0130876526, 0.0005481626,  0.0013482889,  0.0048138603,  0.0007742508,  0.0010990438,  0.0044578823, 
				0.0213519453, -0.0034050515, -0.0020110565, 0.0021831299,  0.0050509793,  0.0042405264,  0.0091022134,  0.0106807724,  0.0039349633, 
				0.0042178528,  0.0050257511,  0.0185917747, 0.0003854073, -0.0052472927,  0.0011107481, -0.0064807054, -0.0096032172,  0.0103662299, 
				0.0011713688,  0.0073811477,  0.0069765832, 0.0085214976, -0.0049973573,  0.0129323883,  0.0085906385, -0.0197146876,  0.0135953952, 
				0.0029944503};

		/*
		double [] rlm = {0.03450307, -0.11994839,  0.28162376,  0.15798554, -7.92325426,  6.91719643,  2.84079377,  0.02558989,  2.98105144, 
				0.07360629,  0.02401240,  0.04833213,  0.03037863,  5.52469013,  2.32648356,  1.13330107,  0.55958626,  1.25183982, 
				0.03458122,  0.08522647, -0.02804902,  0.04307128, -0.21561792, -0.01968803,  0.19378929,  0.16147329,  0.17828597, 
				0.02682765,  0.05478299,  0.05623759,  0.54407872, -0.42090195, -0.17300980, -0.07340830, -0.17470747,  0.01301164, 
				0.05600984, -0.46355952,  0.39415956,  0.16892666, -0.06566154,  0.16153389,  0.06998010,  0.37578270, -0.21391034, 
				-0.17156237, -0.07816966, -0.06824226, -0.14186633, -7.30325992, -2.85664442, -2.03051413, -2.15610028,  0.32507753, 
				1.86068016,  1.64633358,  1.22688387, -0.23816942,  1.01147172,  0.80864178, -0.09900410,  0.48073475,  0.00570288, 
				-0.04639973};*/

		assertTrue(ks.kolmogorovSmirnovTest(monomvn, stan_hs) > 0.00001);
		System.out.println("Draws:");
		for (int i = 0; i < stan_hs.length; i++) {
			System.out.print(stan_hs[i]  + "<=>" + monomvn[i] + ", ");
		}
		System.out.println();
		System.out.println("Monomvn OK!");

		assertTrue(ks.kolmogorovSmirnovTest(mons, stan_hs) > 0.00001);
		System.out.println("Draws:");
		for (int i = 0; i < stan_hs.length; i++) {
			System.out.print(stan_hs[i]  + "<=>" + mons[i] + ", ");
		}
		System.out.println();
		System.out.println("Mons OK!");

		assertTrue(ks.kolmogorovSmirnovTest(java1, stan_hs) > 0.00001);
		System.out.println("Draws:");
		for (int i = 0; i < stan_hs.length; i++) {
			System.out.print(stan_hs[i]  + "<=>" + java1[i] + ", ");
		}
		System.out.println();
		System.out.println("Java 1 OK!");

		/*
		assertTrue(ks.kolmogorovSmirnovTest(rlm, stan_hs) > 0.00001);
		System.out.println("Draws:");
		for (int i = 0; i < stan_hs.length; i++) {
			System.out.print(stan_hs[i]  + "<=>" + rlm[i] + ", ");
		}
		System.out.println();
		System.out.println("R lm OK!");
		*/
	}

}
