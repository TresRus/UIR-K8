package org.mephi.neuralNet;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.mephi.addClasses.MyMath;

public class TrainGA implements Train {
	private NeuroNet net;
	private static Random gen = new Random();
			
	public TrainGA(NeuroNet nnet) {
		net = nnet;
	}

	@Override
	public void trainNet(double[] data) throws Exception {
		trainFirstStep(data);
		
		System.out.printf("GA 0: %f\n", net.countNetFit(data));
		for(int i = 1; i <= 15; ++i) {
			trainStep(data,i);
			System.out.printf("GA %d: %f\n", i, net.countNetFit(data));
		}
	}
	
	@Override
	public void addTrainNet(double[] data) throws Exception {
		for(int i = 10; i <= 20; ++i) {
			trainStep(data,i);
			System.out.printf("GA %d: %f\n", i, net.countNetFit(data));
		}
	}

	private void trainFirstStep(double[] data) throws Exception {
		int index;
		int iterat = net.getwSize();
		
		int pullSize = net.getwSize()/2;
		int minPullSize = net.getwSize()/6;
		if(pullSize < 9)
			pullSize = 9;
		if(minPullSize < 5)
			minPullSize = 5;
		
		double delPerIter = (double) pullSize/ (((double) iterat) * 0.7);
		double toDel = 0.0;
		
		List<double[]> pull = firstPull(data, pullSize);
		double[] pullFit = countPullFit(pullSize, pull, data);
		
		for(int j = 0; j < iterat; ++j)
		{
			toDel += delPerIter;
			
			//List<double[]> cross = crossingover(pull.get(MyMath.secMinInArr(pullFit)),pull.get(MyMath.minInArr(pullFit)));
			List<double[]> cross = crossingover(pull.get(gen.nextInt(pullSize)),pull.get(MyMath.minInArr(pullFit)));
			
			step(cross, pull, pullFit, data, 1);
			
			while(toDel > 1.0)
			{
				if(pullSize == minPullSize)
				{
					toDel = 0.0;
				}
				else
				{
					index = MyMath.maxInArr(pullFit);
					pull.remove(index);
					pullFit = MyMath.replaceFromArray(index,pullFit);
					toDel -= 1.0;
					pullSize -= 1;
				}
			}
			//index = MyMath.minInArr(pullFit);
			//System.out.printf("GA: %f\n", pullFit[index]);
		}
		
		//index = MyMath.minInArr(pullFit);
		//net.setWeights(pull.get(index));
	}
	
	private void trainStep(double[] data, int lv) throws Exception {
		int index;
		int iterat = (int) (net.getwSize()/(2*Math.pow(lv, 0.7)));
		
		int pullSize = net.getwSize()/(2*lv);
		int minPullSize = net.getwSize()/(6*lv);
		if(pullSize < 9)
			pullSize = 9;
		if(minPullSize < 5)
			minPullSize = 5;
		
		double delPerIter = (double) pullSize/ (((double) iterat) * 0.7);
		double toDel = 0.0;
		
		List<double[]> pull = nextPull(data, pullSize, lv);
		double[] pullFit = countPullFit(pullSize, pull, data);
		
		for(int j = 0; j < iterat; ++j)
		{
			toDel += delPerIter;
			
			//List<double[]> cross = crossingover(pull.get(MyMath.secMinInArr(pullFit)),pull.get(MyMath.minInArr(pullFit)));
			List<double[]> cross = crossingover(pull.get(gen.nextInt(pullSize)),pull.get(MyMath.minInArr(pullFit)));
			
			step(cross, pull, pullFit, data, lv);
			
			while(toDel > 1.0)
			{
				if(pullSize == minPullSize)
				{
					toDel = 0.0;
				}
				else
				{
					index = MyMath.maxInArr(pullFit);
					pull.remove(index);
					pullFit = MyMath.replaceFromArray(index,pullFit);
					toDel -= 1.0;
					pullSize -= 1;
				}
			}
			//index = MyMath.minInArr(pullFit);
			//System.out.printf("GA: %f\n", pullFit[index]);
		}
		//index = MyMath.minInArr(pullFit);
		//net.setWeights(pull.get(index));
	}
	
	private void step(List<double[]> cross, List<double[]> pull, double[] pullFit, double[] data, int lv) throws Exception {
		double[] nw;
		double nwf;
		int maxFitIndex;
		
		double mltp = Math.pow(lv, 0.3);
		
		for(int i = 0; i < 3; ++i)
		{
			nw = cross.get(i);
			nw = mutation(nw, 0.03 * mltp, lv);
			net.setWeights(nw);
			nwf = net.countNetFit(data);
			if(Double.isNaN(nwf))
				break;
			maxFitIndex = MyMath.maxInArr(pullFit);
			if(pullFit[maxFitIndex] > nwf)
			{
				pull.remove(maxFitIndex);
				pull.add(maxFitIndex, nw);
				pullFit[maxFitIndex] = nwf;
			}
		}
	}
	
	private double[] mutation(double[] arr, double ch, int lv)
	{
		double[] m = new double[net.getwSize()];
		double r;
		
		double dev = Math.pow(lv, 0.3);
		
		for(int i = 0; i < net.getwSize(); ++i)
		{
			m[i] = arr[i];
			r = gen.nextDouble();
			if(r < ch)
			{
				m[i] += ((gen.nextDouble()-0.5) * 0.0001)/dev;
			}
		}
		
		return m;
	}
	
	private List<double[]> crossingover(double[] p1, double[] p2)
	{
		List<double[]> res = new ArrayList<double[]>();
		double[] c1 = new double[net.getwSize()];
		double[] c2 = new double[net.getwSize()];
		double[] c3 = new double[net.getwSize()];
		double b = gen.nextDouble();
		
		for(int i = 0; i < net.getwSize(); ++i)
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
	
	private double[] countPullFit(int pullSize, List<double[]> pull,double[] data) throws Exception
	{
		double[] pullFit = new double[pullSize];
		double[] weight;
		
		for(int i = 0; i < pullSize; ++i)
		{
			weight = pull.get(i);
			net.setWeights(weight);
			pullFit[i] = net.countNetFit(data);
		}
		
		return pullFit;
	}
	
	private List<double[]> firstPull(double[] data, int size) throws Exception
	{
		List<double[]> pull = new ArrayList<double[]>();
		pull.add(net.getWeights());
		double[] w;
		
		for(int i = 1; i < size; ++i)
		{
			w = new double[net.getwSize()];
			for(int j = 0; j < net.getwSize(); ++j)
			{
				w[j] = (gen.nextDouble() - 0.5) * 1.3;
			}
			
			pull.add(w);
		}
		
		return pull;
	}
	
	private List<double[]> nextPull(double[] data, int size, int lv) throws Exception
	{
		List<double[]> pull = new ArrayList<double[]>();
		
		double[] pw = net.getWeights();
		
		pull.add(pw);
		
		double[] w;
		
		double dev = Math.pow(lv, 0.4);
		
		for(int i = 1; i < size; ++i)
		{
			w = new double[net.getwSize()];
			for(int j = 0; j < net.getwSize(); ++j)
			{
				w[j] = pw[j] + (gen.nextDouble() - 0.5)/dev;
			}
			
			pull.add(w);
		}
		
		return pull;
	}
}
