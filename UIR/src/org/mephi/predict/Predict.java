package org.mephi.predict;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.mephi.addClasses.MyMath;
import org.mephi.neuralNet.*;

public class Predict {

	private static Random gen = new Random();
	
	public static void main(String[] args) {
		try{
			List<double[]> dataIn;
			List<double[]> dataOut;
			NeuroNet net;
			Train tga;
			double[] data;
			double[] test;
			
			//tga.trainNet(data);
			//tbe.trainNet(data);
			
			//net.testRun(test);
			
			dataIn = DataIO.readData("E:\\data.xls");
			
			test = dataIn.get(0);
			data = MyMath.getSubArr(0, test.length - 7, test);
			
			System.out.println("get data");
			
			net = new NeuroNet(0,MyMath.getMaxInArr(test) * 1.3 ,new int[] {30,15,1}, new int[] {1,2,3},30);
			tga = new TrainGA(net);
			
			System.out.println("train...");
			
			tga.trainNet(data);
			tga.addTrainNet(data);
			
			System.out.println("train end");
			
			net.testRun(test);
			
			System.out.println("end");
			
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
