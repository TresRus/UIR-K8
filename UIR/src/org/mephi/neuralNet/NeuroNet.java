package org.mephi.neuralNet;

import java.util.List;

import org.mephi.addClasses.MyMath;

public class NeuroNet {
	private Layer[] layers;
	private double minVal;
	private double devVal;
	private int wSize;
	private int numOfInputs;
	private int numOfOutputs;
	
	public int getwSize() {
		return wSize;
	}

	public int getNumOfInputs() {
		return numOfInputs;
	}

	public int getNumOfOutputs() {
		return numOfOutputs;
	}

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
		numOfInputs = numberOfInputs;
		if(size.length == functions.length)
		{
			wSize = 0;
			int n = numberOfInputs;
			int m = numberOfInputs;
			for(int i = 0; i < size.length; ++i)
			{
				n = m;
				m = size[i];
				wSize += (n+1)*m;
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
			numOfOutputs = out;
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
		numOfInputs = numberOfInputs;
		if(size.length == functions.length)
		{
			wSize = 0;
			int n = numberOfInputs;
			int m = numberOfInputs;
			for(int i = 0; i < size.length; ++i)
			{
				n = m;
				m = size[i];
				wSize += (n+1)*m;
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
			numOfOutputs = out;
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
	
	public double[] predictRun(double[] input, int len) throws Exception
	{
		if(numOfOutputs == 1)
		{
			if(numOfInputs == input.length)
			{
				double[] in = input;
				double out;
				double[] res = new double[len];
				for(int i = 0; i < len; ++i)
				{
					out = runNet(in)[0];
					res[i] = out;
					in = MyMath.getSubArr(1, in.length-1, in);
					in = MyMath.appendToArr(in, out);
				}
				return res;
			}
			else
			{
				throw new Exception("Input length not enough.");
			}
		}
		else
		{
			throw new Exception("Net not accessable for predict.");
		}
	}
	
	//normalization
	
	public double[] normalize(double[] arr)
	{
		double[] res = new double[arr.length];
		for(int i = 0; i < arr.length; ++i)
		{
			res[i] = (arr[i] - minVal)/devVal;
		}
		return res;
	}
	
	public double[] denormalize(double[] arr)
	{
		double[] res = new double[arr.length];
		for(int i = 0; i < arr.length; ++i)
		{
			res[i] = arr[i] * devVal + minVal;
		}
		return res;
	}
	
	//getter and setter of all weights
	
	public double[] getWeights()
	{
		double[] res = new double[wSize];
		int i = 0;
		
		for(Layer l:layers)
		{
			for(Neuron n : l.getNeurons())
			{
				for(double w:n.getWeight())
				{
					res[i] = w;
					++i;
				}
				res[i] = n.getB();
				++i;
			}
		}
		
		return res;
	}
	
	public void setWeights(double[] w) throws Exception
	{
		if(w.length != wSize)
		{
			throw new Exception("Wront weight size.");
		}
		int i = 0;
		
		for(Layer l:layers)
		{
			for(Neuron n : l.getNeurons())
			{
				n.setWeight(MyMath.getSubArr(i,n.getWeight().length,w));
				i += n.getWeight().length;
				n.setB(w[i]);
				++i;
			}
		}
	}
	
	public double countNetFit(double[] date) throws Exception {
		double[] in;
		double[] test;
		double[] out;
		double res = 0.0;
		double add;
		int m = (date.length - getNumOfInputs() - getNumOfOutputs() + 1);
		
		for(int i = 0; i < m; ++i)
		{
			in = MyMath.getSubArr(i,getNumOfInputs(),date);
			test = MyMath.getSubArr(i+getNumOfInputs(),getNumOfOutputs(),date);
			out = runNet(in);
			add = MyMath.countDev(test, out);
			res += add;
		}
		
		return res/m;
	}
	
	public void testRun(double[] date) throws Exception
	{
		double[] in;
		double[] test;
		double[] out;
		
		for(int i = 0; i < (date.length - numOfInputs - numOfOutputs + 1); ++i)
		{
			in = MyMath.getSubArr(i,numOfInputs,date);
			test = MyMath.getSubArr(i+numOfInputs,numOfOutputs,date);
			out = runNet(in);
			for (int j = 0; j < out.length; ++j)
			{
				System.out.printf("out: %10.4f test: %10.4f err in pr : %3.4f \n",out[j],test[j],Math.abs((out[j]-test[j])/test[j]));
			}
		}
	}
}
