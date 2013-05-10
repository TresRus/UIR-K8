package org.mephi.predict;

import java.util.Random;

import org.mephi.neuralNet.*;

public class Predict {

	private static Random gen = new Random();
	
	public static void main(String[] args) {
		try{
			//Neuron n = new Neuron(1,2,0.5);
			//Layer l = new Layer(1,2,5);
			NeuroNet net = new NeuroNet(new int[] {2,2,1}, new int[] {1,2,3},1);
			double[][] tIn = createInput();
			double[][] tOut = testRes(tIn);
			for (int i = 0; i < 40000; ++i)
			{
				for(int j = 0; j < 1; ++j)
				{
					double[] in = tIn[j];
					double[] test = tOut[j];
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
		int m = 1;
		double[][] res = new double[n][m];
		for(int i = 0; i < m; ++i)
		{
			for(int j = 0; j < n; ++j)
			{
				res[j][i] = (gen.nextDouble() - 0.5) * 5;
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
			res[i][0] = in[i][0];
		}
		return res;
	}
}
