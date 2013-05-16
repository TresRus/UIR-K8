package org.mephi.predict;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.mephi.neuralNet.*;

public class Predict {

	private static Random gen = new Random();
	
	public static void main(String[] args) {
		try{
			NeuroNet net = new NeuroNet(0,90100,new int[] {30,15,1}, new int[] {1,2,3},30);
			//double[][] tIn = createInput();
			//double[][] tOut = testRes(tIn);
			double[] date = lineInput(300);
			double[] out;
			
			out = net.trainNetGA(date, 5000);
			
			for (int i = 0; i < out.length; ++i)
			{
				System.out.printf("out: %f test: %f\n",out[i],date[date.length-1]);
			}
			
			net.testRun(date);
			
			
		}
		catch(Exception e)
		{
			System.out.println(e.getMessage());
		}
	}
	
	private static double[] lineInput(int n)
	{
		int x;
		
		double[] res = new double[n];
		for(int i = 0; i < n; ++i)
		{
			x = i + 1;
			res[i] = x*x + Math.log10(x)*1/Math.tan(x);
		}
		
		return res;
	}
}
