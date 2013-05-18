package org.mephi.neuralNet;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuroNet {
	private Layer[] layers;
	private double minVal;
	private double devVal;
	private int wSize;
	private int numOfInputs;
	private int numOfOutputs;
	
	private static Random gen = new Random();
	
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
	
	//Train BE
	
	public double[] trainNetBE(double[] date, int iterat) throws Exception
	{
		double n = 0.0000001;
		double[] input;
		double[] test;
		double[] res = null;
		int m = (date.length - numOfInputs - numOfOutputs + 1);
		
		for(int i = 0; i < iterat; ++i)
		{
			for(int j = 0; j < m; ++j)
			{
				input = getSubArr(j,numOfInputs,date);
				test = getSubArr(j+numOfInputs,numOfOutputs,date);
				res = trainBEStep(input, test, n);
			}
			/*
			f1 = countNetFit(date);
			if(f2 < f1)
			{
				n -= mn;
				if (n <= 0)
				{
					n = an;
					an /= 10;
					mn /= 10;
				}
				if(n <= 0.00000000001)
				{
					n = 0.005;
					an = 0.0001;
					mn = 0.0005;
				}
			}
			else
			{
				n += an;
			}
			f2 = f1;
			*/
			System.out.print("BE: ");
			System.out.println(countNetFit(date));
		}
		System.out.print("BE: ");
		System.out.println(countNetFit(date));
		
		return res;
	}
	
	private double[] trainBEStep(double[] input, double[] testOut, double n) throws Exception
	{
		if(testOut.length == layers[layers.length-1].getSize())
		{
			List<double[]> outs = new ArrayList<double[]>();
			List<double[]> deltas = new ArrayList<double[]>();
			
			double[] out;
			double[] res = normalize(input);
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
			double[] test = normalize(testOut);
			
			delta = new double[layers[layers.length-1].getSize()];
			for(int i = 0; i < delta.length; ++i)
			{
				a = outs.get(layers.length-1)[i];
				delta[i] = a*(1-a)*(test[i] - a);
			}
			deltas.add(delta);
			
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
				}
				deltas.add(delta);
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
	
	//normalization
	
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
				n.setWeight(getSubArr(i,n.getWeight().length,w));
				i += n.getWeight().length;
				n.setB(w[i]);
				++i;
			}
		}
	}
	
	//additional method
	
	private static double[] getSubArr(int startIndex, int leng, double[] sourceArr)
	{
		double[] res = new double[leng];
		
		for(int i = 0; i < leng; ++i)
		{
			res[i] = sourceArr[i + startIndex];
		}
		
		return res;
	}
	
	private static double[] replaceFromArray(int index, double[] sourceArr)
	{
		double[] res = new double[sourceArr.length - 1];
		
		for(int i = 0; i < res.length; ++i)
		{
			if(i < index)
			{
				res[i] = sourceArr[i];
			}
			else
			{
				res[i] = sourceArr[i + 1];
			}
		}
		
		return res;
	}
	
	//Train GA
	
	public double[] trainNetGA(double[] date, int iterat) throws Exception
	{
		double[] res;
		
		double[] nw;
		double nwf;
		int maxFitIndex;
		
		int bestW;
		
		int pullSize = wSize*2;
		int minPullSize = wSize/10;
		if(pullSize < 9)
			pullSize = 9;
		if(minPullSize < 5)
			minPullSize = 5;
		
		double delPerIter = (double) pullSize/ (((double) iterat) * 0.85);
		double toDel = 0.0;
		
		List<double[]> pull = firstPull(date, pullSize);
		double[] pullFit = countPullFit(pullSize, pull, date);
		
		for(int j = 0; j < iterat; ++j)
		{
			toDel += delPerIter;
			
			List<double[]> cross = crossingover(pull.get(secMinInArr(pullFit)),pull.get(minInArr(pullFit)));
			//List<double[]> cross = crossingover(pull.get(gen.nextInt(pullSize)),pull.get(minInArr(pullFit)));
			
			/*
			if( j == (int) (iterat * 0.9))
			{
				for(int i = 0; i < pullSize; ++i)
				{
					double[] w = pull.get(i);
					setWeights(w);
					System.out.printf("# %d : %d ", i, pullSize);
					trainNetBE(date,20);
					w = getWeights();
					pull.remove(i);
					pull.add(i, w);
				}
			}
			*/
			for(int i = 0; i < 3; ++i)
			{
				nw = cross.get(i);
				nw = mutation(nw, 0.03);
				setWeights(nw);
				nwf = countNetFit(date);
				if(Double.isNaN(nwf))
					break;
				maxFitIndex = maxInArr(pullFit);
				if(pullFit[maxFitIndex] > nwf)
				{
					pull.remove(maxFitIndex);
					pull.add(maxFitIndex, nw);
					pullFit[maxFitIndex] = nwf;
				}
			}
			
			while(toDel > 1.0)
			{
				if(pullSize == minPullSize)
				{
					toDel = 0.0;
				}
				else
				{
					maxFitIndex = maxInArr(pullFit);
					pull.remove(maxFitIndex);
					pullFit = replaceFromArray(maxFitIndex,pullFit);
					toDel -= 1.0;
					pullSize -= 1;
				}
			}
			
			bestW = minInArr(pullFit);
			System.out.print("GA: ");
			System.out.println(pullFit[bestW]);
		}
		
		bestW = minInArr(pullFit);
		System.out.println(pullFit[bestW]);
		setWeights(pull.get(bestW));
		
		res = runNet(getSubArr(date.length - numOfInputs - 2,numOfInputs, date));
		
		return res;
	}

	private double[] mutation(double[] arr, double ch)
	{
		double[] m = new double[wSize];
		double min = arr[minInArr(arr)];
		double max = arr[maxInArr(arr)];
		double r;
		
		for(int i = 0; i < wSize; ++i)
		{
			m[i] = arr[i];
			r = gen.nextDouble();
			if(r < ch)
			{
				m[i] = min + (max-min) * gen.nextDouble();
			}
		}
		
		return m;
	}
	
	private List<double[]> crossingover(double[] p1, double[] p2)
	{
		List<double[]> res = new ArrayList<double[]>();
		double[] c1 = new double[wSize];
		double[] c2 = new double[wSize];
		double[] c3 = new double[wSize];
		double b = gen.nextDouble();
		
		for(int i = 0; i < wSize; ++i)
		{
			c1[i] = b * p1[i] + (1-b) * p2[i];
			c2[i] = (1 + b) * p1[i] - b * p2[i];
			c3[i] = - b * p1[i] + (1 + b) * p2[i];
		}
		res.add(c1);
		res.add(c2);
		res.add(c3);
		
		return res;
	}
	
	private double[] countPullFit(int pullSize, List<double[]> pull,double[] date) throws Exception
	{
		double[] pullFit = new double[pullSize];
		double[] weight;
		
		for(int i = 0; i < pullSize; ++i)
		{
			weight = pull.get(i);
			setWeights(weight);
			pullFit[i] = countNetFit(date);
		}
		
		return pullFit;
	}
	
	private List<double[]> firstPull(double[] date, int size) throws Exception
	{
		List<double[]> pull = new ArrayList<double[]>();
		pull.add(getWeights());
		double[] w;
		
		for(int i = 0; i < size; ++i)
		{
			w = new double[wSize];
			for(int j = 0; j < wSize; ++j)
			{
				w[j] = ((gen.nextDouble() * 1.0) - 0.5) * 1.3;
			}
			
			pull.add(w);
		}
		
		return pull;
	}
	
	private double countNetFit(double[] date) throws Exception
	{
		double[] in;
		double[] test;
		double[] out;
		double res = 0.0;
		double add;
		int m = (date.length - numOfInputs - numOfOutputs + 1);
		
		for(int i = 0; i < m; ++i)
		{
			in = getSubArr(i,numOfInputs,date);
			test = getSubArr(i+numOfInputs,numOfOutputs,date);
			out = runNet(in);
			add = countDev(test, out);
			res += add;
		}
		
		return res/m;
	}
	
	private double countDev(double[] arr1,double[] arr2) throws Exception
	{
		if(arr1.length == arr2.length)
		{
			double dev = 0.0;
			for(int i = 0 ; i < arr1.length; ++i)
			{
				dev += Math.abs(arr1[i] - arr2[i]);
			}
			
			return dev;
		}
		else
		{
			throw new Exception("Wrong arr size");
		}
	}
	
	private int minInArr(double[] arr)
	{
		double min = arr[0];
		int n = 0;
		
		for(int i = 0; i < arr.length; ++i)
		{
			if(min > arr[i])
			{
				min = arr[i];
				n = i;
			}
		}
		
		return n;
	}
	
	private int maxInArr(double[] arr)
	{
		double max = arr[0];
		int n = 0;
		
		for(int i = 0; i < arr.length; ++i)
		{
			if(max < arr[i])
			{
				max = arr[i];
				n = i;
			}
		}
		
		return n;
	}
	
	private int secMinInArr(double[] arr)
	{
		double min = arr[0];
		int n = 0;
		int s = 0;
		
		for(int i = 0; i < arr.length; ++i)
		{
			if(min > arr[i])
			{
				min = arr[i];
				s = n;
				n = i;
			}
		}
		
		return s;
	}
	
	public void testRun(double[] date) throws Exception
	{
		double[] in;
		double[] test;
		double[] out;
		
		for(int i = 0; i < (date.length - numOfInputs - numOfOutputs + 1); ++i)
		{
			in = getSubArr(i,numOfInputs,date);
			test = getSubArr(i+numOfInputs,numOfOutputs,date);
			out = runNet(in);
			for (int j = 0; j < out.length; ++j)
			{
				System.out.printf("out: %f test: %f err in pr : %f \n",out[j],test[j],Math.abs((out[j]-test[j])/test[j]));
			}
		}
	}
}
