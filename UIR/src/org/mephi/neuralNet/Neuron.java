package org.mephi.neuralNet;

import java.util.Random;

public class Neuron {
	private double[] weight;
	private int fun;
	
	public double[] getWeight() {
		return weight;
	}

	public Neuron(int f, int numberOfInputs)
	{
		fun = f;
		Random gen = new Random();
		weight = new double[numberOfInputs];
		for(int i = 0; i < weight.length; ++i)
		{
			weight[i] = (gen.nextDouble() * 2.0) - 1.0;
		}
	}
	
	public Neuron(int f, int numberOfInputs, double value)
	{
		fun = f;
		weight = new double[numberOfInputs];
		for(int i = 0; i < weight.length; ++i)
		{
			weight[i] = value;
		}
	}
	
	public double RunNeuron(double[] input) throws Exception
	{
		if(input.length == weight.length)
		{
			double funInput = 0.0;
			for(int i = 0; i < weight.length; ++i)
			{
				funInput += input[i]*weight[i];
			}
			return applyFun(funInput);
		}
		else
		{
			throw new Exception("Wrong input size.");
		}
	}
	
	private double applyFun(double value) throws Exception
	{
		double res = 0.0;
		switch (fun) {
		case 1: res = gausFun(value);
			break;
		case 2: res = sigmFun(value);
			break;
		case 3: res = lineFun(value);
			break;
		default: throw new Exception("Unkown function");
		}
		return res;
	}
	
	private double lineFun(double value)
	{
		return value;
	}
	
	private double sigmFun(double value)
	{
		return (1/(1 + java.lang.Math.pow(java.lang.Math.E, -value)));
	}
	
	private double gausFun(double value)
	{
		double m = 0.0;
		double d = 1.0;
		double res = 0.0;
		res = 1/(d * java.lang.Math.pow(2 * java.lang.Math.PI, 0.5));
		res *= java.lang.Math.pow(java.lang.Math.E, - ( java.lang.Math.pow(value - m,2) / (2 * java.lang.Math.pow(d,2)) ) );
		return res;
	}
}
