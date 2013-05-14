package org.mephi.predict;

import java.util.Random;

import org.mephi.neuralNet.*;

public class Predict {

	private static Random gen = new Random();
	
	public static void main(String[] args) {
		try{
			NeuroNet net = new NeuroNet(0,15,new int[] {15,6,1}, new int[] {1,2,3},10);
			//double[][] tIn = createInput();
			//double[][] tOut = testRes(tIn);
			double[] date = lineInput();
			for (int i = 0; i < 5000; ++i)
			{
				for(int j = 0; j < 1; ++j)
				{
					double[] in = getSubArr(j,10,date);
					double[] test = getSubArr(j+10,1,date);
					double[] out = net.trainNet(in, test, 0.1);
					System.out.printf("%d . %d:\n",i,j);
					for (int k = 0; k < out.length; ++k)
					{
						System.out.printf("out: %f test: %f\n",out[k],test[k]);
					}
				}
			}
		}
		catch(Exception e)
		{
			System.out.println(e.getMessage());
		}
	}
	
	private static double[][] createInput()
	{
		int n = 10;
		int m = 2;
		double[][] res = new double[n][m];
		for(int i = 0; i < m; ++i)
		{
			for(int j = 0; j < n; ++j)
			{
				res[j][i] = (i + j - 6) / 5;
			}
		}
		return res;
	}
	
	private static double[][] testRes(double[][] in)
	{
		int n = 10;
		double[][] res = new double[n][1];
		for(int i = 0; i < n; ++i)
		{
			res[i][0] = in[i][0] + in[i][1];
		}
		return res;
	}
	
	private static double[] lineInput()
	{
		int n = 21;
		int x;
		
		double[] res = new double[n];
		for(int i = 0; i < n; ++i)
		{
			x = i +2;
			res[i] = x;
		}
		
		return res;
	}
	
	private static double[] getSubArr(int startIndex, int leng, double[] sourceArr)
	{
		double[] res = new double[leng];
		
		for(int i = 0; i < leng; ++i)
		{
			res[i] = sourceArr[i + startIndex];
		}
		
		return res;
	}
}
