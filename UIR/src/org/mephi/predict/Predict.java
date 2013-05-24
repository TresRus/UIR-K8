package org.mephi.predict;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.mephi.neuralNet.*;

public class Predict {

	private static Random gen = new Random();
	
	public static void main(String[] args) {
		try{
			List<double[]> dataIn;
			List<double[]> dataOut;
			NeuroNet net = new NeuroNet(0,300,new int[] {30,15,1}, new int[] {1,2,3},30);
			Train tga = new TrainGA(net);
			Train tbe = new TrainBE(net);
			//double[][] tIn = createInput();
			//double[][] tOut = testRes(tIn);
			double[] date = lineInput(300);
			double[] test = lineTest(300);
			
			//tga.trainNet(date);
			//tbe.trainNet(date);
			
			//net.testRun(test);
			
			dataIn = DataIO.readData("C:\\data.xls");
			
		}
		catch(Exception e)
		{
			System.out.println(e.getMessage());
		}
	}
	
	private static double[] lineInput(int n)
	{
		double x;
		
		double[] res = new double[n];
		for(int i = 0; i < n; ++i)
		{
			x = i + 20;
			res[i] = 100 * Math.sin(x/10) +  10 * Math.sin(x/5) * Math.cos(x/2) + 150;
		}
		
		return res;
	}
	
	private static double[] lineTest(int n)
	{
		double x;
		int m = n + 10;
		
		double[] res = new double[m];
		for(int i = 0; i < m; ++i)
		{
			x = i + 20;
			res[i] = 100 * Math.sin(x/10) +  10 * Math.sin(x/5) * Math.cos(x/2) + 150;
		}
		
		return res;
	}
}
