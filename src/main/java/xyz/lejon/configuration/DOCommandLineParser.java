package xyz.lejon.configuration;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;

public class DOCommandLineParser implements CommandLineParserUtil {
	
	CommandLine parsedCommandLine;
	CommandLineParser parser;
	String configFn = null;
	String comment  = null;

	public DOCommandLineParser(String [] args) throws ParseException {
		
		parser = new PosixParser();

		Options options = new Options();
		options.addOption( "idims", "initial_dims", true, 
				"scale the dataset to initial dims with PCA before running " + DOConfiguration.PROGRAM_NAME);
		options.addOption( "nolbls", "no_labels",   false, 
				"The dataset does not contain any labels (if not set labels are assumed to be in the first column)" );
		options.addOption( "lblf", "label_file",    true, 
				"Separate input file with dataset labels, one label per row, must contain at least as many rows as in the dataset. Extra labels will be thrown away" );
		options.addOption( "tlblf", "trainingset_label_file",    true, 
				"Separate input file with trainingset labels, one label per row, must contain at least as many rows as in the trainingset. Extra labels will be thrown away. Labels in the test set must match those in the trainingset" );
		options.addOption( "icept", "intercept",    true, 
				"Use intercept, 0=false, 1=true, default is true" );
		options.addOption( "lblcolno", "label_column_no",    true, 
				"If labels are not in first column, this option gives the index of the label column" );
		options.addOption( "lblcolnme", "label_column_name",    true, 
				"If labels are not in first column, this option gives the name of the label column. Requires headers in the dataset" );
		options.addOption( "na", "na_string",   true, 
				"Dataset can contain N/A, the given string is parsed as N/A in the dataset" );
		options.addOption( "nohdr", "no_headers",   false, 
				"If set, won't try to read a first row of column headers / names" );
		options.addOption( "log", "scale_log",      false, 
				"Scale the dataset by first taking the log of each datapoint (keeping zeros) " );
		options.addOption( "norm", "normalize",     false, 
				"Normalize the data by subtracting the mean and dividing by the stdev (this is done after eventual log) " );
		options.addOption( "iter", "iterations",    true, 
				"How many iterations to run, default is " + DOConfiguration.ITERATIONS_DEFAULT);
		options.addOption( "plt", "plot",           false, 
				"Plot the resulting dataset " );
		options.addOption( "shw", "show",           false, 
				"Show displays the tabular data of a data frame in a gui window " );
		options.addOption( "dn",  "drop_names",      true, 
				"drop column names. Takes a list of names (Example: \"Customer Name,Comment,Id\") representing the cloumn names to drop. This is done AFTER any drop_column!" );
		options.addOption( "dc",  "drop_columns",    true, 
				"drop column no's. Takes a list of integers (Example: \"1,2,8,11\") representing the cloumns to drop" );
		options.addOption( "sep", "separator",      true, 
				"column separator ',' , ';' , '\\t' (',' per default). '\\t' denotes tab" );
		options.addOption( "dbl", "double_default", false, 
				"use Double as number format (Long is default but even with Long default, numbers with decimals will still be parsed as Double)" );
		options.addOption( "trsp", "transpose", false, 
				"transpose the dataset first" );
		options.addOption( "out", "output_file",    true, 
				"Save the result to the given filename" );
		options.addOption( "dbg", "debug", true, 
				"use debugging " );
		options.addOption( "cm", "comment", true, 
				"a comment ot be added to the logfile " );
		options.addOption( "ds", "dataset", true, 
				"filename of dataset file" );
		options.addOption( "ts", "topics", true, 
				"number of topics" );
		options.addOption( "a", "alpha", true, 
				"uniform alpha prior" );
		options.addOption( "b", "beta", true, 
				"uniform beta prior" );
		options.addOption( "i", "iterations", true, 
				"number of sample iterations" );
		options.addOption( "batch", "batches", true, 
				"the number of batches to split the data in" );
		options.addOption( "r", "rare_threshold", true, 
				"the number of batches to split the data in" );
		options.addOption( "ti", "topic_interval", true, 
				"topic interval" );
		options.addOption( "sd", "start_diagnostic", true, 
				"start diagnostic" );
		options.addOption( "sch", "scheme", true, 
				"sampling scheme " );
		options.addOption( "cf", "run_cfg", true, 
				"full path to the RunConfiguration file " );

		HelpFormatter formatter = new HelpFormatter();

		// Try to parse the command line
		try {
			parsedCommandLine = parser.parse( options, args );
		} catch (org.apache.commons.cli.ParseException e) {
			System.out.println(DOConfiguration.PROGRAM_NAME  + ": Could not parse command line due to:  " + e.getMessage());
			System.out.println("Args where:");
			for (int i = 0; i < args.length; i++) {
				System.out.print(args[i] + ", ");
			}
			formatter.printHelp(DOConfiguration.PROGRAM_NAME  + " [options] <csv file>", options );
			System.exit(-1);
		}
		
		if( parsedCommandLine.hasOption( "cm" ) ) {
			comment = parsedCommandLine.getOptionValue( "comment" );
		}
		if( parsedCommandLine.hasOption( "cf" ) ) {
			configFn = parsedCommandLine.getOptionValue( "run_cfg" );
		}

		// Make sure we got a CSV file or a config file name
		if(parsedCommandLine.getArgs().length==0 && configFn == null) {
			System.out.println("No CSV file given and no config file...");
			formatter.printHelp(DOConfiguration.PROGRAM_NAME  + " [options] <csv file>", options );
			System.exit(255);
		}	
	}
	
	public String getConfigFn() {
		return configFn;
	}
	
	public String getComment() {
		return comment;
	}
	
	public CommandLine getParsedCommandLine() {
		return parsedCommandLine;
	}

}
