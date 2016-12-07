package xyz.lejon.configuration;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import joinery.DataFrame;
import joinery.DataFrame.NumberDefault;
import joinery.impl.Conversion;

import org.apache.commons.configuration.ConfigurationException;

import xyz.lejon.utils.DataSet;
import xyz.lejon.utils.FileUtils;
import xyz.lejon.utils.LoggingUtils;
import xyz.lejon.utils.MatrixOps;
import xyz.lejon.utils.PrincipalComponentAnalysis;

public class ParsedDOConfiguration extends SubConfig implements DOConfiguration, Configuration {

	private static final long serialVersionUID = 1L;

	DOCommandLineParser commandlineParser = null;
	LoggingUtils logger;
	String comment        = null;
	int initial_dims      = -1;
	int iterations        = 100;
	boolean hasHeader     = true;
	String naString       = null;
	boolean hasLabels     = true;
	String trainingset_label_fn = null;
	String testset_label_fn     = null;
	boolean scale_log     = false;
	boolean normalize     = false;
	boolean addIntercept  = true;
	boolean doPlot        = false;
	boolean doSave        = false;
	int label_col_no      = 1;
	String label_col_name = null;
	int id_col_no         = -1;
	String id_col_name    = null;

	DataSet trainingSet;
	DataSet testSet; 

	String sep = ",";

	public ParsedDOConfiguration(DOCommandLineParser cp) throws ConfigurationException {
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

		if (parsedCommandLine.hasOption( "betas_output_file" )) {
			doSave = true;
		}
		
		sep = parsedCommandLine.hasOption( "separator" ) ? (String) parsedCommandLine.getOptionValue("separator").trim() : ",";

		if(!(sep.equals(",") || sep.equals(";") || sep.equals("\\t"))) {
			System.out.println("Only the separators ',' , ';' or '\\t' is currently supported...");
			System.exit(255);
		}

	}

	public ParsedDOConfiguration(String path) throws ConfigurationException {
		super(path);
		whereAmI = path;
		setDefaultListDelimiter(',');
	}

	public LoggingUtils getLoggingUtil() {
		if(logger==null) throw new IllegalArgumentException("You havent initialized the Logger before usage");
		return logger;
	}

	public void setLoggingUtil(LoggingUtils logger) {
		this.logger = logger;
	}
	
	public String getExperimentOutputDirectory(String defaultDir) {
		String dir = getStringProperty("experiment_out_dir");
		if(dir != null && dir.endsWith("/")) dir = dir.substring(0,dir.length()-1);
		return (dir == null) ? defaultDir : dir;
	}

	public String getDatasetFilename() {
		 return getStringProperty("testset");
	}

	public String getTrainingsetFilename() {
		return getStringProperty("testset");
	}

	public Integer getNoIterations(int defaultValue) {
		return getInteger("iterations",defaultValue);
	}

	public Integer getStartDiagnostic(int defaultValue) {
		return getInteger("start_diagnostic",defaultValue);
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
		String key = "debug";
		return getBooleanProperty(key);
	}

	public boolean getPrintPhi() {
		String key = "print_phi";
		return getBooleanProperty(key);
	}

	public boolean getMeasureTiming() {
		String key = "measure_timing";
		return getBooleanProperty(key);
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
	
	public static boolean haveFilename(String filename) {
		return filename!=null && filename.trim().length()>0;
	}

	/**
	 * If the labels are in the DataFrame extract them as numbers to 'ys' 
	 * and strings to 'plotLabels' and remove that column from the DataFrame
	 * Also build a mapping from numeric <-> string labels
	 * 
	 * @param df
	 * @param plotLabels
	 * @param ys
	 * @param labelToId
	 * @param idToLabels
	 * @return
	 * @throws IOException
	 */
	protected LabelRes extractLabels(DataFrame<Object> df, String [] plotLabels, int [] ys,
			Map<String,Integer> labelToId,	Map<Integer, String> idToLabels) throws IOException {
		// We extract label column before dropping
		int lblcnt = 0;

		int idx = 0;
		if(hasLabels && df != null) {
			plotLabels = new String[df.length()];
			List<Object> col = null;
			if(label_col_name!=null){
				System.out.println("Using labels from colum index: " + label_col_name);
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
				ys[idx++] = labelToId.get(lbl.toString());
			}

			// Now drop the label column
			if(label_col_name!=null){
				df = df.drop(label_col_name);
			} else {
				df = df.drop(label_col_no);
			}
		}

		// Load labels from external file if they are not in the data set
		if(!hasLabels && haveFilename(trainingset_label_fn)) {
			System.out.println("Loading labels from:" + trainingset_label_fn);
			String [] readLabels = FileUtils.readLines(new File(trainingset_label_fn));
			for (int i = 0; i < readLabels.length; i++) {
				String lbl = readLabels[i];
				if(labelToId.get(lbl.toString())==null) {
					labelToId.put(lbl.toString(),lblcnt);
					idToLabels.put(lblcnt++,lbl.toString());
				}
				ys[idx++] = labelToId.get(lbl.toString());
			}

			// Got too many labels, remove the extra ones
			if(df != null && readLabels.length > df.length()) {
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
			csvFilename = getAdditionalDatasetTestFilename();
		}
		
		int [] ys = null;
		String [] plotLabels = null;
		Map<String,Integer> labelToId = trainingSet.getLabelToId();
		Map<Integer, String> idToLabels = trainingSet.getIdToLabels();

		if(csvFilename==null) {
			if(haveFilename(testset_label_fn)) {
				plotLabels = loadTestLabelsFromFile(null, ys, labelToId);
				testSet = new DataSet(null, null, ys, plotLabels, labelToId, idToLabels, null, null);
			} else {
				testSet = null;
			}
			return null;
		} else {
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
			ys = new int[noRows];

			// IF the labels are in the DataFrame extract them as numbers to 'ys' 
			// and strings to 'plotLabels' and remove that column from the DataFrame
			// Also build a mapping from numeric <-> string labels
			int idx = 0;
			if(hasLabels) {
				plotLabels = new String[df.length()];
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
					ys[idx++] = labelToId.get(lbl.toString());
				}

				// Now drop the label column
				if(label_col_name!=null){
					df = df.drop(label_col_name);
				} else {
					df = df.drop(label_col_no);
				}
			}

			// Load labels from external file if they are not in the data set
			if(!hasLabels && haveFilename(testset_label_fn)) {
				plotLabels = loadTestLabelsFromFile(df, ys, labelToId);
			}
			
			ExtractDropResult res = extractAndDropIds(df, id_col_no, id_col_name);
			String [] ids = res.ids;
			df = res.df;

			df = dropColumns(df);
			System.out.println("Testset types:" + df.types());
			System.out.println(df.head(10));

			testSet = new DataSet(df, null, ys, plotLabels, labelToId, idToLabels, null, ids);
			return df; 
		}
	}
	
	protected static class ExtractDropResult {
		public String [] ids;
		public DataFrame<Object> df;
		public ExtractDropResult(String[] ids, DataFrame<Object> df) {
			this.ids = ids;
			this.df = df;
		}
	}
	
	static String[] extractIds(DataFrame<Object> df, int id_col_no, String id_col_name) {
		List<Object> idCol;
		if(id_col_name!=null) {
			idCol = df.col(id_col_name);
		} else {
			idCol = df.col(id_col_no);
		}
		String [] result = new String[idCol.size()];
		for (int i = 0; i < result.length; i++) {
			if(idCol.get(i)==null) {
				throw new IllegalArgumentException("Cannot handle 'null' in Id column!");
			} else {				
				result[i] = idCol.get(i).toString(); 
			}
		}
		return result;
	}
	
	protected ExtractDropResult extractAndDropIds(DataFrame<Object> df, int id_col_no, String id_col_name) {
		String [] ids = null;
		if(id_col_no>=0 || id_col_name!=null) {				
			ids = extractIds(df, id_col_no, id_col_name);
			System.out.println("Extracted ids...");
			// Now drop the id column
			if(id_col_name!=null){
				df = df.drop(id_col_name);
				System.out.println("Dropping col: " + id_col_name);
			} else {
				df = df.drop(id_col_no);
				System.out.println("Dropping col: " + id_col_no);
			}
		}
		return new ExtractDropResult(ids, df);
	}

	String[] loadTestLabelsFromFile(DataFrame<Object> df, int[] ys, Map<String, Integer> labelToId)
			throws IOException {
		int idx = 0;
		String[] plotLabels;
		System.out.println("Loading testset labels from:" + testset_label_fn);
		String [] readLabels = FileUtils.readLines(new File(testset_label_fn));

		// This is the case when we are loading the labels from an external file
		// and we dont know beforehand how many labels the file contains
		if(ys==null) {
			ys = new int[readLabels.length];
		}
		for (int i = 0; i < readLabels.length; i++) {
			String lbl = readLabels[i];
			if(labelToId.get(lbl.toString())==null) {
				throw new IllegalArgumentException("Found label ("+ lbl.toString() +") in test set that was not in training set, cannot handle that!");
			}
			ys[idx++] = labelToId.get(lbl.toString());
		}

		// Got too many labels, remove the extra ones
		if(df!=null && readLabels.length > df.length()) {
			plotLabels = Arrays.copyOf(readLabels, df.length());
		} else {
			plotLabels = readLabels;
		}
		return plotLabels;
	}
	
	@Override
	public DataSet loadTestSet() throws IOException {
		if(testSet==null) return null;
		DataFrame<Object> df = testSet.getOrigData();

		String refCatString = getStringProperty("reference_categories");
		Map<String,String> refCats = parseReferenceCategories(refCatString);
		double [][] xs = Conversion.toModelMatrix(df, 0.0,trainingSet.getOrigData(), false, refCats);
		
		if(scale_log) xs = MatrixOps.log(xs, true);
		if(normalize) xs = MatrixOps.centerAndScale(xs);
		if(addIntercept) xs = MatrixOps.addIntercept(xs);

		if(xs[0].length!=trainingSet.getX()[0].length) {
			throw new IllegalArgumentException("Trainingset and testset doesn not have the same number of covariates " 
					+ xs[0].length + " != " + trainingSet.getX()[0].length 
					+ ". This can occur if there are covariates with nominal attributes in the test set which are not in the trainingset");
		}
		
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
		
        HashValueComparator bvc =  new HashValueComparator(testSet.labelToId);
        TreeMap<String,Integer> sorted_map = new TreeMap<String,Integer>(bvc);
        sorted_map.putAll(testSet.labelToId);
		
		for(String k : sorted_map.keySet()) {
			System.out.print(k + " => " + testSet.labelToId.get(k) + ", ");
		}
		System.out.println();

		testSet.setX(xs);
		testSet.setPca(trainingSet.getPca());
		return testSet;	
	}

	public DataSet loadTrainingSet() throws IOException {
		// The reason all these ugly extractions are here is 
		// because we have sub-configs. We cannot do this at construction
		// since no sub-config is set at that stage... but it is at
		// this point :(
		
		if(getInt("id_column_no", -1)!=-1) {
			id_col_no = getInt("id_column_no", -1) - 1; // Id col is 1-indexed
		}
		if (parsedCommandLine.hasOption( "id_column_no" )) {
			label_col_no = Integer.parseInt(parsedCommandLine.getOptionValue("id_column_no").trim());
			label_col_no--; // Label col is 1 indexed
		}
		if(getStringProperty("id_column_name")!=null) {
			id_col_name = getStringProperty("id_column_name");
		}
		if (parsedCommandLine.hasOption( "id_column_name" )) {
			id_col_name = parsedCommandLine.getOptionValue( "id_column_name" );
		}
		if(getInt("label_column_no", -1)!=-1) {
			label_col_no = getInt("label_column_no", -1) - 1; // Label col is 1-indexed
		}
		if (parsedCommandLine.hasOption( "label_column_no" )) {
			label_col_no = Integer.parseInt(parsedCommandLine.getOptionValue("label_column_no").trim());
			label_col_no--; // Label col is 1 indexed
		}
		if(getStringProperty("label_column_name")!=null) {
			label_col_name = getStringProperty("label_column_name");
		}
		if (parsedCommandLine.hasOption( "label_column_name" )) {
			label_col_name = parsedCommandLine.getOptionValue( "label_column_name" );
		}
		if (parsedCommandLine.hasOption( "no_labels" )) {
			hasLabels = false;
		}
		if(getStringProperty("training_label_file")!=null) {
			trainingset_label_fn = getStringProperty("training_label_file").trim();
		}
		if (parsedCommandLine.hasOption( "training_label_file" )) {
			trainingset_label_fn = parsedCommandLine.getOptionValue("training_label_file").trim();
		}
		if (parsedCommandLine.hasOption( "scale_log" )) {
			System.out.println("Log transforming dataset...");
			scale_log = true;
		}
				
		if(configHasProperty("normalize")) {			
			normalize = getBooleanProperty("normalize");			
		}
		// Command line overrides config...
		if (parsedCommandLine.hasOption( "normalize" )) {
			System.out.println("Normalizing dataset...");
			normalize = true;
		}
		
		if(normalize) System.out.println("Normalizing dataset...");
		else System.out.println("NOT Normalizing dataset...");
		
		if (getBoolean("no_headers", true)) {
			hasHeader = false;
		}
		if (getStringProperty("betas_output_file")!=null) {
			doSave = true;
		}
		
		if(configHasProperty("intercept")) {
			addIntercept = getBooleanProperty("intercept");	
			if(addIntercept) System.out.println("Using intercept...");
		}
		//Command line overrides config...
		if (parsedCommandLine.hasOption( "intercept" )) {
			int icept = Integer.parseInt(parsedCommandLine.getOptionValue("intercept").trim());
			if(icept!=0 && icept!=1 ) 
				throw new IllegalArgumentException("Intercept must be 0 (false) or 1 (true),  " + icept + " is not a legal value");
			addIntercept = icept == 1;
			if(addIntercept) System.out.println("Using intercept...");
		}
		
		String csvFilename = null;
		if(parsedCommandLine.getArgs().length>0) {
			csvFilename = parsedCommandLine.getArgs()[0];
		} else {
			csvFilename = getAdditionalDatasetTrainFilename();
			if(csvFilename !=null && csvFilename.length()==0) {
				csvFilename = null;
			}
		}

		System.out.println("NA string is: " + naString) ;

		final List<DataFrame<Object>> frames = new ArrayList<>();
		DataFrame<Object> df = null;
		int [] ys;
		String [] plotLabels = null;
		Map<String,Integer> labelToId = new HashMap<>();
		Map<Integer, String> idToLabels = new HashMap<>();
		
		// Don't load dataset if we have no valid filename, just load the labels
		if(csvFilename==null) {
			if(!haveFilename(trainingset_label_fn)) {
				throw new IllegalArgumentException("No csv file and no label file, cannot load trainingdata");
			}
			ys = null;
			LabelRes extracted = extractLabels(df, plotLabels, ys, labelToId, idToLabels);
			plotLabels = extracted.plotLables;
			trainingSet = new DataSet(null, null, ys, plotLabels, labelToId, idToLabels, null, null);
			loadTestDataFrame();
		} else {
			if(parsedCommandLine.hasOption( "double_default" ) ) {
				df = DataFrame.readCsv(csvFilename,sep,NumberDefault.DOUBLE_DEFAULT, naString, hasHeader);
				frames.add(df);
			} else {
				System.out.println("Loading from: " + csvFilename);
				df = DataFrame.readCsv(csvFilename,sep,NumberDefault.LONG_DEFAULT, naString, hasHeader);
				frames.add(df);
			}

			System.out.println("Loaded CSV with: " + df.length()+ " rows and " + df.size() +" columns.");

			if (parsedCommandLine.hasOption( "transpose" )) {
				df = df.transpose();
			}
			if(getInteger("initial_dims", -1)!=-1) {
				initial_dims = getInteger("initial_dims", -1);
			}
			if (parsedCommandLine.hasOption( "initial_dims" )) {
				initial_dims = Integer.parseInt(parsedCommandLine.getOptionValue("initial_dims").trim());
				if(initial_dims>df.size()) 
					throw new IllegalArgumentException("Requested more initial dims " + initial_dims + "than available in the dataset: " + df.size());
			}

			int noRows = df.length();
			ys = new int[noRows];

			LabelRes extracted = extractLabels(df, plotLabels, ys, labelToId, idToLabels);
			plotLabels = extracted.plotLables;
			df = extracted.df;
			
			ExtractDropResult res = extractAndDropIds(df, id_col_no, id_col_name);
			String [] ids = res.ids;
			df = res.df;
			
			df = dropColumns(df);

			System.out.println("Dataset types:" + df.types());
			System.out.println(df.head(10));
			//System.out.println(df.summary());

			//DataFrame<Double> ddf = df.cast(Double.class);
			//double [][] xstmp = ddf.fillna(0.0).toArray(double[][].class);
			//xs = MatrixOps.makeDesignMatrix(xstmp);

			// At this point we must save the training set so far, now we must do  
			// the first part of loading the test set (which uses this part of the 
			// training set), this complicated scheme is due to the fact that we 
			// must ensure that all variables in the test set is also in the 
			// training set and vice-versa.
			trainingSet = new DataSet(df, null, ys, plotLabels, labelToId, idToLabels, null, ids);
			loadTestDataFrame();

			DataFrame<Number> mmdf; 

			double [][] xs;
			String refCatString = getStringProperty("reference_categories");
			Map<String,String> refCats = parseReferenceCategories(refCatString);
			if(haveTestSet())
				mmdf = Conversion.toModelMatrixDataFrame(df,testSet.getOrigData(),false, refCats, null);
			else 
				mmdf = Conversion.toModelMatrixDataFrame(df,null,false, refCats, null);

			System.out.println(mmdf.columns());
			xs = mmdf.fillna(0.0).toArray(double[][].class);
			
			if(scale_log) xs = MatrixOps.log(xs, true);
			if(normalize) xs = MatrixOps.centerAndScale(xs);
			if(addIntercept) xs = MatrixOps.addIntercept(xs);

			System.out.println("Trainingset width before PCA: " + xs[0].length);

			boolean didPca = false;
			PrincipalComponentAnalysis pca = null;
			if(initial_dims >0 && xs[0].length>initial_dims) {
				pca = new PrincipalComponentAnalysis();
				xs = pca.pca(xs, initial_dims);
				didPca = true;
			}

			int noCovariates = xs[0].length;
			int noClasses = labelToId.size();

			System.out.println("Have " + noRows + " rows");
			System.out.println("Have " + noCovariates + " covariates");
			System.out.print("Have " + noClasses + " classes:");

			HashValueComparator bvc =  new HashValueComparator(labelToId);
			TreeMap<String,Integer> sorted_map = new TreeMap<String,Integer>(bvc);
			sorted_map.putAll(labelToId);

			for(String k : sorted_map.keySet()) {
				System.out.print(k + " => " + labelToId.get(k) + ", ");
			}
			System.out.println();

			if (parsedCommandLine.hasOption( "show" )) {
				for (int i = 0; i < frames.size(); i++) {
					frames.get(0).show();				 
				}
			}

			String [] names = new String[xs[0].length];
			if(didPca) {
				for (int i = 0; i < names.length; i++) {
					names[i] = "PC" + i;
				}
			} else {
				Set<Object> colnames = mmdf.columns();
				int cIdx = 0;
				if(addIntercept) {
					names[cIdx++] = "Intercept";
				}
				Iterator<Object> cNamesIterator = colnames.iterator();
				while (cIdx < names.length) {
					names[cIdx++] = cNamesIterator.next().toString();
				}
			}
			
			trainingSet.setColnamesX(names);
			trainingSet.setX(xs);
			trainingSet.setPca(pca);
		}
		return trainingSet;
	}
	
	class HashValueComparator implements Comparator<String> {

	    Map<String, Integer> base;
	    public HashValueComparator(Map<String, Integer> base) {
	        this.base = base;
	    }
 
	    public int compare(String a, String b) {
	        if (base.get(a) >= base.get(b)) {
	            return 1;
	        } else {
	            return -1;
	        }
	    }
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

	public boolean getSaveBetas() {
		String key = "save_betas";
		return getBooleanProperty(key);
	}

	public String betasOutputFn() {
		return getStringProperty("betas_output_file");
	}

	public String betaSamplesOutputFn() {
		return getStringProperty("beta_samples_output_file");
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
		return trainingset_label_fn;
	}

	public Integer labelCol() {
		return label_col_no;
	}

	@Override
	public int getLag() {
		return getInteger("lag",1);
	}

	@Override
	public int getBurnIn() {
		Integer bi = getInteger("burn_in",0);
		if(bi<0||bi>99) throw new IllegalArgumentException("Illegal burn_in value (" + bi + "), it must be 0 <= bi <= 99");
		return bi;	
	}

	@Override
	public String getSamplerClass(String modelDefault) {
		String configProperty = getStringProperty("sampler_class");
		return (configProperty == null) ? modelDefault : configProperty;
	}
	
	protected String getAdditionalDatasetTestFilename() {
		return getStringProperty("additional_covariates_test");
	}
	
	protected String getAdditionalDatasetTrainFilename() {
		return getStringProperty("additional_covariates_train");
	}

	@Override
	public boolean saveBetaSamples() {
		return (getStringProperty("save_beta_samples")!=null) && 
				(getStringProperty("save_beta_samples").equalsIgnoreCase("true") || getStringProperty("save_beta_samples").equals("1"));
	}

	public static Map<String, String> parseReferenceCategories(String cat_ref) {
		if(cat_ref==null || cat_ref.length()<1) return null;
		Map<String, String> result = new HashMap<String, String>();
		String [] pairs = cat_ref.trim().split(",");

		for (int i = 0; i < pairs.length; i++) {
			String [] keyVal = pairs[i].split("=>");
			if(keyVal.length!=2) throw new IllegalArgumentException("Found fauly spec: " + pairs[i]);
			String key = keyVal[0].trim();
			if(key.length()<1) throw new IllegalArgumentException("Found empty key");
			String value = keyVal[1].trim();
			if(value.length()<1) throw new IllegalArgumentException("Found empty value");
			if(isQuoted(key)) {
				key = key.substring(1, key.length()-1);
			}
			if(isQuoted(value)) {
				value = value.substring(1, value.length()-1);
			}
			result.put(key, value);
		}
		return result;
	}

	protected static boolean isQuoted(String key) {
		return (key.startsWith("\"") && key.endsWith("\"")) || (key.startsWith("'") && key.endsWith("'"));
	}
	
	public boolean getUseIntercept() {
		return addIntercept;
	}
	
	public boolean getNormalize() {
		if (parsedCommandLine.hasOption( "normalize" )) {
			return true;
		}
		if(configHasProperty("normalize")) {
			return getBooleanProperty("normalize");
		}
		return normalize;
	}
}
