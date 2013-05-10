package org.mephi.neuralNet;

public class NeuroNet {
	private Layer[] layers;
	
	public NeuroNet(int[] size,int[] functions, int numberOfInputs) throws Exception
	{
		if(size.length == functions.length)
		{
			layers = new Layer[size.length];
			int in = numberOfInputs;
			int out = 0;
			for(int i = 0; i < size.length; ++i)
			{
				out = size[i];
				layers[i] = new Layer(functions[i],in,out);
				in = out;
			}
		}
		else
		{
			throw new Exception("Wrong number of size or functions");
		}
	}
	
	public NeuroNet(int[] size,int[] functions, int numberOfInputs, double weightVal) throws Exception
	{
		if(size.length == functions.length)
		{
			layers = new Layer[size.length];
			int in = numberOfInputs;
			int out = 0;
			for(int i = 0; i < size.length; ++i)
			{
				out = size[i];
				layers[i] = new Layer(functions[i],in,out,weightVal);
				in = out;
			}
		}
		else
		{
			throw new Exception("Wrong number of size or functions");
		}
	}
	
	public double[] runNet(double[] input) throws Exception
	{
		double[] res = input;
		for(Layer l : layers)
		{
			res = l.runLayer(res);
		}
		return res;
	}
}
