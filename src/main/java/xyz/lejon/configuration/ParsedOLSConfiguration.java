package xyz.lejon.configuration;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

import joinery.DataFrame;
import joinery.DataFrame.NumberDefault;
import joinery.impl.Conversion;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.configuration.HierarchicalINIConfiguration;

import xyz.lejon.utils.FileUtils;
import xyz.lejon.utils.LoggingUtils;
import xyz.lejon.utils.MatrixOps;
import xyz.lejon.utils.PrincipalComponentAnalysis;
import xyz.lejon.utils.RegressionDataSet;

public class ParsedOLSConfiguration extends HierarchicalINIConfiguration implements OLSConfiguration, Configuration {

	private static final long serialVersionUID = 1L;

	protected String subConfName = null;
	protected OLSCommandLineParser commandlineParser = null;
	protected String whereAmI;
	LoggingUtils logger;
	protected CommandLine parsedCommandLine;
	protected String comment  = null;
	protected String configFn = null;
	protected String fullPath = null;
	
	protected int initial_dims   = -1;
	protected int iterations     = 100;
	protected boolean hasHeader  = true;
	protected String naString    = null;
	protected boolean hasLabels  = false;
	protected String label_fn    = null;
	protected String trainingset_label_fn    = null;
	protected boolean scale_log  = false;
	protected boolean normalize  = false;
	protected boolean doPlot     = false;
	protected String output_fn   = null;
	protected boolean doSave     = false;
	protected int label_col_no   = 0;
	protected String label_col_name     = null;
	protected String sep;	
	protected int response_col_no = 0;
	protected String response_col_name = null;

	protected RegressionDataSet trainingSet;
	protected RegressionDataSet testSet;

	protected boolean addIntercept = false;

	public ParsedOLSConfiguration() {

	}

	public ParsedOLSConfiguration(OLSCommandLineParser cp) throws ConfigurationException {
		super(cp.getConfigFn());
		setDefaultListDelimiter(',');
		commandlineParser = cp;
		whereAmI = cp.getConfigFn();
		parsedCommandLine = cp.getParsedCommandLine();

		if( parsedCommandLine.hasOption( "cm" ) ) {
			comment = parsedCommandLine.getOptionValue( "comment" );
		}
		if( parsedCommandLine.hasOption( "cf" ) ) {
			configFn = parsedCommandLine.getOptionValue( "run_cfg" );
		}

		if (parsedCommandLine.hasOption( "iterations" )) {
			iterations = Integer.parseInt(parsedCommandLine.getOptionValue("iterations").trim());
		}

		if (parsedCommandLine.hasOption( "no_headers" )) {
			hasHeader = false;
		}

		if (parsedCommandLine.hasOption( "na_string" )) {
			naString = parsedCommandLine.getOptionValue("na_string").trim();
		}

		if (parsedCommandLine.hasOption( "plot" )) {
			doPlot = true;
		}

		if (parsedCommandLine.hasOption( "output_file" )) {
			output_fn = parsedCommandLine.getOptionValue("output_file");
			doSave = true;
		}

		sep = parsedCommandLine.hasOption( "separator" ) ? (String) parsedCommandLine.getOptionValue("separator").trim() : ",";

		if(!(sep.equals(",") || sep.equals(";") || sep.equals("\\t"))) {
			System.out.println("Only the separators ',' , ';' or '\\t' is currently supported...");
			System.exit(255);
		}

	}

	public ParsedOLSConfiguration(String path) throws ConfigurationException {
		super(path);
		whereAmI = path;
		setDefaultListDelimiter(',');
	}

	/* (non-Javadoc)
	 * @see configuration.LDAConfiguration#getLoggingUtil()
	 */

	public LoggingUtils getLoggingUtil() {
		if(logger==null) throw new IllegalArgumentException("You havent initialized the Logger before usage");
		return logger;
	}

	/* (non-Javadoc)
	 * @see configuration.LDAConfiguration#setLoggingUtil(utils.LoggingUtils)
	 */

	public void setLoggingUtil(LoggingUtils logger) {
		this.logger = logger;
	}

	/* (non-Javadoc)
	 * @see configuration.LDAConfiguration#activateSubconfig(java.lang.String)
	 */

	public void activateSubconfig(String subConfName) {
		boolean foundIt = false;
		String [] configs = super.getStringArray("configs");
		for( String cfg : configs ) {
			cfg = cfg.trim();
			if( subConfName.equals(cfg) ) {
				foundIt = true;
			}
		}
		if( !foundIt ) {
			throw new IllegalArgumentException("No such configuration: " + subConfName);
		}
		this.subConfName = subConfName; 
	}

	/* (non-Javadoc)
	 * @see configuration.LDAConfiguration#forceActivateSubconfig(java.lang.String)
	 */

	public void forceActivateSubconfig(String subConfName) {
		this.subConfName = subConfName; 
	}

	/* (non-Javadoc)
	 * @see configuration.LDAConfiguration#getActiveSubConfig()
	 */

	public String getActiveSubConfig() {
		return subConfName;
	}


	/**
	 * First check if we are in a subconfig and have the key, if so return it. 
	 * Else check if the key is in the global scope, if so return it.
	 * Else if we are in a subconfig scope, return the subconfig key otherwise 
	 * just return the key.
	 * @param key
	 * @return
	 */
	private String translateKey(String key) {
		if(subConfName!=null && containsKey(subConfName + "." + key)) {
			return subConfName + "." + key;
		} else if (containsKey(key)) {
			return key;
		} else if(subConfName!=null) {
			return subConfName + "." + key;
		} else {
			return key;
		}
	}

	public String [] getStringArrayProperty(String key) {
		return trimStringArray(super.getStringArray(translateKey(key)));
	}

	public int [] getIntArrayProperty(String key, int [] defaultValues) {
		String [] ints = super.getStringArray(translateKey(key));
		if(ints==null || ints.length==0) {
			//throw new IllegalArgumentException("Could not find any int array for key:" + translateKey(key));
			return defaultValues;
		}
		int [] result = new int[ints.length];
		for (int i = 0; i < ints.length; i++) {
			result[i] = Integer.parseInt(ints[i].trim());
		}
		return result;
	}

	public double [] getDoubleArrayProperty(String key) {
		String [] ints = super.getStringArray(translateKey(key));
		if(ints==null || ints.length==0) { 
			throw new IllegalArgumentException("Could not find any double array for key:" 
					+ translateKey(key)); 
		}
		double [] result = new double[ints.length];
		for (int i = 0; i < ints.length; i++) {
			result[i] = Double.parseDouble(ints[i].trim());
		}
		return result;
	}


	protected String [] trimStringArray(String [] toTrim) {
		for (int i = 0; i < toTrim.length; i++) {
			toTrim[i] = toTrim[i].trim();
		}
		return toTrim;
	}


	public String getStringProperty(String key) {
		if(parsedCommandLine.hasOption(key) && parsedCommandLine.getOptionValue(key)!=null) {
			return parsedCommandLine.getOptionValue(key);
		} else {
			// This hack lets us have "," in strings
			String strProp = "";
			Object prop = super.getProperty(translateKey(key));
			if(prop instanceof java.util.List) {
				@SuppressWarnings("unchecked")
				List<String> propParts = (List<String>) prop;
				for (String string : propParts) {
					strProp += string + ",";
				}
				strProp = strProp.substring(0, strProp.length()-1);
			} else {
				strProp = (String) prop;
			}
			return strProp;
		}
	}

	/* (non-Javadoc)
	 * @see configuration.LDAConfiguration#getConfProperty(java.lang.String)
	 */

	public Object getConfProperty(String key) {
		return super.getProperty(translateKey(key));
	}

	@Override
	public boolean getBoolean(String key)
	{
		return super.getBoolean(translateKey(key));
	}

	@Override
	public boolean getBoolean(String key, boolean defaultValue)
	{
		try {
			return getBoolean(key);
		} catch (Exception e) {
			return false;
		}
	}
	/* (non-Javadoc)
	 * @see org.apache.commons.configuration.AbstractHierarchicalFileConfiguration#setProperty(java.lang.String, java.lang.Object)
	 */

	public void setProperty(String key, Object value) {
		super.setProperty(translateKey(key), value);
	}

	/* (non-Javadoc)
	 * @see configuration.LDAConfiguration#getSubConfigs()
	 */

	public String[] getSubConfigs() {
		return trimStringArray(super.getStringArray("configs"));
	}


	public Integer getInteger(String key, Integer defaultValue) {
		if(parsedCommandLine.hasOption(key) && parsedCommandLine.getOptionValue(key)!=null) {
			return Integer.parseInt(parsedCommandLine.getOptionValue(key.trim()));
		} else {
			return super.getInteger(translateKey(key),defaultValue);
		}
	}


	public Double getDouble(String key, Double defaultValue) {
		if(parsedCommandLine.hasOption(key) && parsedCommandLine.getOptionValue(key)!=null) {
			return Double.parseDouble(parsedCommandLine.getOptionValue(key.trim()));
		} else {
			return super.getDouble(translateKey(key),defaultValue);
		}
	}

	/* (non-Javadoc)
	 * @see configuration.LDAConfiguration#whereAmI()
	 */

	public String whereAmI() {
		return whereAmI;
	}

	/* (non-Javadoc)
	 * @see configuration.LDAConfiguration#getDatasetFilename()
	 */

	public String getDatasetFilename() {
		return getStringProperty("dataset");
	}

	public String getTrainingsetFilename() {
		return getStringProperty("testset");
	}

	public Integer getNoIterations(int defaultValue) {
		return getInteger("iterations",defaultValue);
	}

	/**
	 * @param seedDefault
	 * @return the seed set in the config file or the LSB of the current time if set to -1
	 */

	public int getSeed(int seedDefault) {
		int seed = getInteger("seed",seedDefault);
		if(seed==0) {seed=(int)System.currentTimeMillis();};
		return seed;
	}


	public boolean getDebug() {
		return (getStringProperty("debug")!=null) && 
				(getStringProperty("debug").equalsIgnoreCase("true") || getStringProperty("debug").equals("1"));
	}

	public String getOption(String key) {
		return parsedCommandLine.getOptionValue(key);
	}

	public void setConfigFn(String configFn) {
		this.configFn = configFn;
	}

	public String getFullPath() {
		return fullPath;
	}

	public void setFullPath(String fullPath) {
		this.fullPath = fullPath;
	}

	public String getComment() {
		return comment;
	}

	public void setComment(String comment) {
		this.comment = comment;
	}

	public String toString() {
		return "--comment=" + getComment() + " --run_config=" + configFn; 
	}

	static class LabelRes {
		DataFrame<Object> df;
		String [] plotLables;
		public LabelRes(DataFrame<Object> df, String[] plotLables) {
			super();
			this.df = df;
			this.plotLables = plotLables;
		}
	}

	protected LabelRes extractLabels(DataFrame<Object> df, String [] plotLabels,
			Map<String,Integer> labelToId,	Map<Integer, String> idToLabels) throws IOException {
		// We extract label column before dropping
		int lblcnt = 0;

		// IF the labels are in the DataFrame extract them as numbers to 'ys' 
		// and strings to 'plotLabels' and remove that column from the DataFrame
		// Also build a mapping from numeric <-> string labels
		if(hasLabels) {
			plotLabels = new String[df.length()];
			int idx = 0;
			List<Object> col = null;
			if(label_col_name!=null){
				System.out.println("Using labels from colum index: " + (label_col_name+1));
				col = df.col(label_col_name);
			} else {
				System.out.println("Using labels from colum name: " + label_col_no);
				col = df.col(label_col_no);
			}
			for (Object lbl : col) {
				plotLabels[idx] = lbl.toString();
				if(labelToId.get(lbl.toString())==null) {
					labelToId.put(lbl.toString(),lblcnt);
					idToLabels.put(lblcnt++,lbl.toString());
				}
			}

			// Now drop the label column
			if(label_col_name!=null){
				df = df.drop(label_col_name);
			} else {
				df = df.drop(label_col_no);
			}
		}

		// Load labels from external file if they are not in the data set
		if(!hasLabels && label_fn!=null) {
			System.out.println("Loading labels from:" + label_fn);
			String [] readLabels = FileUtils.readLines(new File(label_fn));
			for (int i = 0; i < readLabels.length; i++) {
				String lbl = readLabels[i];
				if(labelToId.get(lbl.toString())==null) {
					labelToId.put(lbl.toString(),lblcnt);
					idToLabels.put(lblcnt++,lbl.toString());
				}

			}

			// Got too many labels, remove the extra ones
			if(readLabels.length > df.length()) {
				plotLabels = Arrays.copyOf(readLabels, df.length());
			} else {
				plotLabels = readLabels;
			}
		}
		return new LabelRes(df,plotLabels);
	}
	
	boolean haveTestSet() {
		return parsedCommandLine.getArgs().length>1 || getTrainingsetFilename()!=null;
	}
	
	protected DataFrame<Object> loadTestDataFrame() throws IOException {
		String csvFilename = null;
		if(parsedCommandLine.getArgs().length>1) {
			csvFilename = parsedCommandLine.getArgs()[1];
		} else {
			csvFilename = getTrainingsetFilename();
		}
		if(csvFilename==null) return null;

		final List<DataFrame<Object>> frames = new ArrayList<>();

		DataFrame<Object> df = null;
		if(parsedCommandLine.hasOption( "double_default" ) ) {
			df = DataFrame.readCsv(csvFilename,sep,NumberDefault.DOUBLE_DEFAULT, naString, hasHeader);
			frames.add(df);
		} else {
			df = DataFrame.readCsv(csvFilename,sep,NumberDefault.LONG_DEFAULT, naString, hasHeader);
			frames.add(df);
		}

		System.out.println("Loaded CSV with: " + df.length()+ " rows and " + df.size() +" columns.");

		if (parsedCommandLine.hasOption( "transpose" )) {
			df = df.transpose();
		}

		int noRows = df.length();

		if (parsedCommandLine.hasOption( "transpose" )) {
			df = df.transpose();
		}
		double [] ys = new double[noRows];
		
		List<Object> theYs;
		if(response_col_name!=null) {
			theYs = df.col(response_col_name);
			df = df.drop(response_col_name);
		} else {
			theYs = df.col(response_col_no);
			df = df.drop(response_col_no);
		}
		int ycnt = 0;
		for(Object yi : theYs) {
			ys[ycnt++] = (Double) yi;
		}
		
		String [] plotLabels = null;
		Map<String,Integer> labelToId = getTrainingSet().getLabelToId();
		Map<Integer, String> idToLabels = getTrainingSet().getIdToLabels();

		// IF the labels are in the DataFrame extract them as numbers to 'ys' 
		// and strings to 'plotLabels' and remove that column from the DataFrame
		// Also build a mapping from numeric <-> string labels
		if(hasLabels) {
			plotLabels = new String[df.length()];
			int idx = 0;
			List<Object> col = null;
			if(label_col_name!=null){
				col = df.col(label_col_name);
			} else {
				col = df.col(label_col_no);
			}
			for (Object lbl : col) {
				plotLabels[idx] = lbl.toString();
				if(labelToId.get(lbl.toString())==null) {
					throw new IllegalArgumentException("Found label ("+ lbl.toString() +") in test set that was not in training set, cannot handle that!");
				}
			}

			// Now drop the label column
			if(label_col_name!=null){
				df = df.drop(label_col_name);
			} else {
				df = df.drop(label_col_no);
			}
		}

		// Load labels from external file if they are not in the data set
		if(!hasLabels && trainingset_label_fn!=null) {
			System.out.println("Loading labels from:" + trainingset_label_fn);
			String [] readLabels = FileUtils.readLines(new File(trainingset_label_fn));
			for (int i = 0; i < readLabels.length; i++) {
				String lbl = readLabels[i];
				if(labelToId.get(lbl.toString())==null) {
					throw new IllegalArgumentException("Found label ("+ lbl.toString() +") in test set that was not in training set, cannot handle that!");
				}
			}

			// Got too many labels, remove the extra ones
			if(readLabels.length > df.length()) {
				plotLabels = Arrays.copyOf(readLabels, df.length());
			} else {
				plotLabels = readLabels;
			}
		}

		df = dropColumns(df);
		System.out.println("Trainingset types:" + df.types());
		System.out.println(df.head(10));
		
		testSet = new RegressionDataSet(df, null, ys, plotLabels, labelToId, idToLabels, null, addIntercept);
		return df; 
	}
	
	@Override
	public RegressionDataSet loadTestSet() throws IOException {
		if(testSet==null) return null;
		DataFrame<Object> df = testSet.getOrigData();
		
		DataFrame<Number> mmdf = Conversion.toModelMatrixDataFrame(df, trainingSet.getOrigData(), false, null, null);
		
		double [][] xs = mmdf.fillna(0.0).toArray(double[][].class);

		if(scale_log) xs = MatrixOps.log(xs, true);
		if(normalize) xs = MatrixOps.centerAndScale(xs);

		// If we PCA'd the training set we use the same transform on the test set
		PrincipalComponentAnalysis pca = trainingSet.getPca();
		if(initial_dims >0 && xs[0].length>initial_dims) {
			xs = pca.translateToSpace(xs);
		}

		int noCovariates = xs[0].length;
		int noClasses = testSet.labelToId.size();

		System.out.println("Test set has " + testSet.getOrigData().length() + " rows");
		System.out.println("Test set has " + noCovariates + " covariates");
		System.out.print("Test set has " + noClasses + " classes:");
		for(String k : testSet.labelToId.keySet()) {
			System.out.print(k + " => " + testSet.labelToId.get(k) + ", ");
		}
		System.out.println();

		testSet.setTransformedData(mmdf);
		testSet.setX(xs);
		testSet.setPca(trainingSet.getPca());
		return testSet;	
	}

	public RegressionDataSet loadTrainingSet() throws IOException {
		String csvFilename = null;
		if(parsedCommandLine.getArgs().length>0) {
			csvFilename = parsedCommandLine.getArgs()[0];
		} else {
			csvFilename = getDatasetFilename();
		}
		
		if(getStringProperty("separator")!=null) {
			sep = getStringProperty("separator");
			System.out.println("Setting separator to '"+ sep + "'");
		}		

		final List<DataFrame<Object>> frames = new ArrayList<>();

		try {
			hasHeader = !getBoolean("no_headers");
		} catch (Exception e) {
		}
		
		if (getStringProperty("output_file")!=null) {
			output_fn = getStringProperty("output_file");
			doSave = true;
		}

		System.out.println("NA string is: " + naString) ;

		DataFrame<Object> df = null;
		if(getStringProperty( "double_default" ) != null) {
			df = DataFrame.readCsv(csvFilename,sep,NumberDefault.DOUBLE_DEFAULT, naString, hasHeader);
			frames.add(df);
		} else {
			df = DataFrame.readCsv(csvFilename,sep,NumberDefault.LONG_DEFAULT, naString, hasHeader);
			frames.add(df);
		}

		System.out.println("Loaded CSV with: " + df.length()+ " rows and " + df.size() +" columns.");

		if (getStringProperty( "transpose" )!=null) {
			df = df.transpose();
		}
		if(getInteger("initial_dims", -1)!=-1) {
			initial_dims = getInteger("initial_dims", -1);
		}
		if(initial_dims>df.size()) 
			throw new IllegalArgumentException("Requested more initial dims " + initial_dims + "than available in the dataset: " + df.size());
		if(getInteger("label_column_no", -1)!=-1) {
			label_col_no = getInt("label_column_no", -1) - 1; // Label col is 1-indexed
		}
		if(getStringProperty("label_column_name")!=null) {
			label_col_name = getStringProperty("label_column_name");
		}
		if (getStringProperty( "label_column_name" )!=null) {
			label_col_name = parsedCommandLine.getOptionValue( "label_column_name" );
		}
		if(getStringProperty("response_column_no")!=null) {
			response_col_no = getInt("response_column_no");
			response_col_no--; // Cols in DataFrame is 1 indexed
		}
		if (getStringProperty( "response_column_no" )!=null) {
			response_col_no = Integer.parseInt(parsedCommandLine.getOptionValue("response_column_no").trim());
			response_col_no--; // Label col is 1 indexed
		}
		if(getStringProperty("response_column_name")!=null) {
			response_col_name = getStringProperty("response_column_name");
		}
		if (getStringProperty( "no_labels" )!=null) {
			hasLabels = false;
		}
		if (getStringProperty( "label_file" )!=null) {
			label_fn = parsedCommandLine.getOptionValue("label_file").trim();
		}
		if (getStringProperty( "training_label_file" )!=null) {
			trainingset_label_fn = getStringProperty("training_label_file").trim();
		}
		if (getStringProperty( "scale_log" )!=null) {
			System.out.println("Log transforming dataset...");
			scale_log = true;
		}
		if (getStringProperty( "normalize" )!=null) {
			System.out.println("Normalizing dataset...");
			normalize = true;
		}
		
		try{
			addIntercept = getBoolean("intercept");
		}
		catch(NoSuchElementException e) {} // Use default if not specified
		
		if(addIntercept) System.out.println("Using intercept...");

		int noRows = df.length();
		
		System.out.println("Dataset types:" + df.types());
		System.out.println(df.head(10));
		//System.out.println(df.summary());

		double [] ys = new double[noRows];
		List<Object> theYs;
		if(response_col_name!=null) {
			theYs = df.col(response_col_name);
			df = df.drop(response_col_name);
		} else {
			theYs = df.col(response_col_no);
			df = df.drop(response_col_no);
		}
		int ycnt = 0;
		for(Object yi : theYs) {
			ys[ycnt++] = (Double) yi;
		}

		String [] plotLabels = null;
		Map<String,Integer> labelToId = new HashMap<>();
		Map<Integer, String> idToLabels = new HashMap<>();

		LabelRes extracted = extractLabels(df, plotLabels, labelToId, idToLabels);
		plotLabels = extracted.plotLables;
		df = extracted.df;

		df = dropColumns(df);

		//DataFrame<Double> ddf = df.cast(Double.class);
		//double [][] xstmp = ddf.fillna(0.0).toArray(double[][].class);
		//xs = MatrixOps.makeDesignMatrix(xstmp);
		
		// At this point we must save the training set so far, now we must do  
		// the first part of loading the test set (which uses this part of the 
		// training set), this complicated scheme is due to the fact that we 
		// must ensure that all variables in the test set is also in the 
		// training set and vice-versa.
		trainingSet = new RegressionDataSet(df, null, ys, plotLabels, labelToId, idToLabels, null, addIntercept);
		loadTestDataFrame();

		double [][] xs;
		DataFrame<Number> mmdf;
		if(haveTestSet()) {
			mmdf = Conversion.toModelMatrixDataFrame(df, testSet.getOrigData(), false, null, null);
			xs = mmdf.fillna(0.0).toArray(double[][].class);
		}
		else { 
			mmdf = Conversion.toModelMatrixDataFrame(df, null, false, null, null);
			xs = mmdf.fillna(0.0).toArray(double[][].class);
		}
		
		if(scale_log) xs = MatrixOps.log(xs, true);
		if(normalize) xs = MatrixOps.centerAndScale(xs);
		if(addIntercept) xs = MatrixOps.addIntercept(xs);
		
		System.out.println("Trainingset width before PCA: " + xs[0].length);
		
		PrincipalComponentAnalysis pca = null;
		if(initial_dims >0 && xs[0].length>initial_dims) {
			pca = new PrincipalComponentAnalysis();
			xs = pca.pca(xs, initial_dims);
		}

		int noCovariates = xs[0].length;
		int noClasses = labelToId.size();

		System.out.println("Have " + noRows + " rows");
		if(addIntercept)
			System.out.println("Have " + noCovariates + " covariates (including one added for intercept) ");
		else 
			System.out.println("Have " + noCovariates + " covariates (not intercept added)");
		
		System.out.print("Have " + noClasses + " classes:");
		for(String k : labelToId.keySet()) {
			System.out.print(k + " => " + labelToId.get(k) + ", ");
		}
		System.out.println();

		if (parsedCommandLine.hasOption( "show" )) {
			for (int i = 0; i < frames.size(); i++) {
				frames.get(0).show();				 
			}
		}
		
		trainingSet.setTransformedData(mmdf);
		trainingSet.setX(xs);
		trainingSet.setPca(pca);
		return trainingSet;
	}

	DataFrame<Object> dropColumns(DataFrame<Object> df) {
		// Now process any drop column arguments, it makes sense to
		// drop the integer indexed first since named ones are 
		// independent of index and thus we can do those after
		// and get the drops correct in case of both index and
		// named drops

		String [] colIdxs = getStringArrayProperty("drop_columns");
		System.out.println("Dropping Idxs: " + Arrays.asList(colIdxs));
		if( parsedCommandLine.hasOption( "drop_columns" ) ) {
			colIdxs = 	parsedCommandLine.getOptionValue("drop_columns").split(",");
		}
		if (colIdxs != null && colIdxs.length > 0) {
			List<Integer> dropIdxs = new ArrayList<>();
			for (int i = 0; i < colIdxs.length; i++) {
				String colVal = colIdxs[i].trim();
				if(colVal.length()>0) dropIdxs.add(Integer.parseInt(colVal));
			}
			Integer [] idxs = dropIdxs.toArray(new Integer[0]);
			if(dropIdxs.size()>0) df = df.drop(idxs);
		}

		String [] colNames = getStringArrayProperty("drop_names");
		if( parsedCommandLine.hasOption( "drop_names" ) ) {
			colNames = 	parsedCommandLine.getOptionValue("drop_names").split(",");
		}
		System.out.println("Dropping: " + Arrays.asList(colNames));
		if (colNames != null && colNames.length > 0) {
			for (String colName : colNames) {
				String trimmedName = colName.trim();
				if(trimmedName.length()>0) df = df.drop(trimmedName);				
			}
		}
		return df;
	}

	public boolean scaleLog() {
		return scale_log;
	}

	public int getIterations() {
		return iterations;
	}

	public boolean doSave() {
		return doSave;
	}

	public String outputFn() {
		return output_fn;
	}

	public boolean doPlot() {
		return doPlot;
	}

	public boolean normalize() {
		return normalize;
	}

	public boolean hasLabels() {
		return hasLabels;
	}

	public Object labelFn() {
		return label_fn;
	}

	public Integer labelCol() {
		return label_col_no;
	}

	@Override
	public double[][] getX() {
		return trainingSet.getX();
	}

	@Override
	public double[] getY() {
		return trainingSet.getY();
	}

	@Override
	public String[] getLabels() {
		return trainingSet.getLabels();
	}

	@Override
	public Map<String, Integer> getLabelMap() {
		return trainingSet.getLabelToId();
	}

	@Override
	public Map<Integer, String> getIdMap() {
		return trainingSet.getIdToLabels();
	}

	@Override
	public int getLag() {
		return getInteger("lag",1);
	}

	@Override
	public int getBurnIn() {
		return getInteger("burn_in",0);	
	}

	@Override
	public RegressionDataSet getTestSet() {
		return testSet;
	}

	@Override
	public RegressionDataSet getTrainingSet() {
		return trainingSet;
	}

	@Override
	public String getSamplerClass(String modelDefault) {
		String configProperty = getStringProperty("sampler_class");
		return (configProperty == null) ? modelDefault : configProperty;
	}
	
	@Override
	public boolean getUseIntercept() {
		return addIntercept;
	}
}
