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
			Train tbe;
			FilterKalman f;
			double[] data;
			double[] fdata;
			double[] test;
			double maxValue;
			
			//tga.trainNet(data);
			//tbe.trainNet(data);
			
			//net.testRun(test);
			
			dataIn = DataIO.readData("E:\\data.xls");
			
			test = dataIn.get(2);
			data = MyMath.getSubArr(0, test.length - 7, test);
			//data = MyMath.getSubArr(0, 110, test);
			
			System.out.println("get data");
			
			maxValue = MyMath.getMaxInArr(test) * 1.3;
			System.out.println(maxValue);
			
			f = new FilterKalman(500, maxValue * 0.03);
			System.out.println("filter data");
			
			fdata = f.filter(data);
			
			System.out.println("filter data");
			
			for(int i = 0; i < data.length; ++i)
			{
				System.out.printf("date: %9.4f fdata: %9.4f noize in pr : %3.4f \n",data[i],fdata[i],Math.abs((data[i]-fdata[i])/data[i]));
			}
			
			net = new NeuroNet(0,maxValue,new int[] {30,10,1}, new int[] {1,2,3},30);
			tga = new TrainGA(net);
			tbe = new TrainBE(net);
			
			System.out.println("train...");
			
			tga.trainNet(fdata);
			//tbe.trainNet(fdata);
			tga.addTrainNet(fdata);
			
			System.out.println("train end");
			
			//net.testRun(fdata);
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
