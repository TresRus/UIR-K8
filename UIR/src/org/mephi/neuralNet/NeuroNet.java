package org.mephi.neuralNet;

import java.util.ArrayList;
import java.util.List;

public class NeuroNet {
	private Layer[] layers;
	private double minVal;
	private double devVal;
	private int wSize;
	
	public Layer[] getLayers() {
		return layers;
	}

	public void setLayers(Layer[] layers) {
		this.layers = layers;
	}

	public NeuroNet(double min, double max, int[] size,int[] functions, int numberOfInputs) throws Exception
	{
		minVal = min;
		devVal = max - min;
		if(size.length == functions.length)
		{
			wSize = 0;
			int n = numberOfInputs;
			int m = numberOfInputs;
			for(int i = 0; i < size.length; ++i)
			{
				n = m;
				m = size[i];
				wSize += n*m;
			}
			
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
	
	public NeuroNet(double min, double max, int[] size,int[] functions, int numberOfInputs, double weightVal) throws Exception
	{
		minVal = min;
		devVal = max - min;
		if(size.length == functions.length)
		{
			wSize = 0;
			int n = numberOfInputs;
			int m = numberOfInputs;
			for(int i = 0; i < size.length; ++i)
			{
				n = m;
				m = size[i];
				wSize += n*m;
			}
			
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
		double[] res = normalize(input);
		for(Layer l : layers)
		{
			res = l.runLayer(res);
		}
		return denormalize(res);
	}
	
	public double[] trainNet(double[] in, double[] testArr, double n) throws Exception
	{
		if(testArr.length == layers[layers.length-1].getSize())
		{
			List<double[]> outs = new ArrayList<double[]>();
			List<double[]> deltas = new ArrayList<double[]>();
			
			double[] out;
			double[] res = normalize(in);
			for( double e : res)
			{
				//System.out.printf("%f \n", e);
			}
			for(Layer l : layers)
			{
				out = new double[l.getSize()];
				res = l.runLayer(res);
				for(int i = 0; i < out.length; ++i)
				{
					out[i] = res[i];
				}
				outs.add(out);
			}
			
			double[] delta;
			double a;
			double[] test = normalize(testArr);
			
			delta = new double[layers[layers.length-1].getSize()];
			for(int i = 0; i < delta.length; ++i)
			{
				a = outs.get(layers.length-1)[i];
				delta[i] = a*(1-a)*(test[i] - a);
				//System.out.printf("delta %d : %f \n", i, delta[i]);
			}
			deltas.add(delta);
			//countNewWeight(layers[layers.length-1],delta,outs.get(layers.length-1),n);
			
			double[] prevDelta;
			for(int i = layers.length-2; i >= 0; --i)
			{
				prevDelta = delta;
				delta = new double[layers[i].getSize()];
				double sum;
				for(int j = 0; j < layers[i].getSize();++j)
				{
					sum = 0.0;
					for(int k = 0; k < layers[i+1].getSize(); ++k)
					{
						sum += prevDelta[k] * layers[i+1].getNeurons()[k].getWeight()[j];
					}
					a = outs.get(i)[j];
					delta[j] = a*(1-a)*sum;
					//System.out.printf("delta %d %d : %f \n", i, j, delta[i]);
				}
				deltas.add(delta);
				//countNewWeight(layers[i],delta,outs.get(i),n);
			}
			
			for(int i = 0; i < layers.length; ++i)
			{
				countNewWeight(layers[layers.length-1-i],deltas.get(i),outs.get(layers.length-1-i),n);
			}
			
			return denormalize(res);
		}
		else
		{
			throw new Exception("Wrong test array size");
		}
	}
	
	private void countNewWeight(Layer l, double[] delta, double[] out, double n)
	{
		double dw;
		double[] w;
		for(int i = 0; i < l.getSize(); ++i)
		{
			w = l.getNeurons()[i].getWeight();
			dw = n * delta[i] * out[i];
			for(int j = 0; j < w.length; ++j)
			{
				w[j] = w[j] + dw;
			}
			l.getNeurons()[i].setWeight(w);
		}
	}
	
	private double[] normalize(double[] arr)
	{
		double[] res = new double[arr.length];
		for(int i = 0; i < arr.length; ++i)
		{
			res[i] = (arr[i] - minVal)/devVal;
		}
		return res;
	}
	
	private double[] denormalize(double[] arr)
	{
		double[] res = new double[arr.length];
		for(int i = 0; i < arr.length; ++i)
		{
			res[i] = arr[i] * devVal + minVal;
		}
		return res;
	}
}
