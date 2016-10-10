package xyz.lejon.utils;

import javax.swing.JFrame;

import org.math.plot.FrameView;
import org.math.plot.Plot2DPanel;
import org.math.plot.PlotPanel;
import org.math.plot.plots.ColoredScatterPlot;
import org.math.plot.plots.ScatterPlot;

public class ClassificationResultPlot {
	
	public static void plot2D(String[] labels, double[][] Y) {
		Plot2DPanel plot = new Plot2DPanel();
		if(labels != null) {
			ColoredScatterPlot scatterPlot = new ColoredScatterPlot("Result", Y, labels);
			plot.plotCanvas.addPlot(scatterPlot);
		} else {
			ScatterPlot dataPlot = new ScatterPlot("Data", PlotPanel.COLORLIST[0], Y);
			plot.plotCanvas.addPlot(dataPlot);

		}
		plot.plotCanvas.setNotable(true);
		plot.plotCanvas.setNoteCoords(true);

		FrameView plotframe = new FrameView(plot);
		plotframe.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		plotframe.setVisible(true);
	}
	
}
