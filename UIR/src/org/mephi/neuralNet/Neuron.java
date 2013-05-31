package org.mephi.neuralNet;

import java.util.Random;

public class Neuron {
	/**
	 * @uml.property  name="weight"
	 */
	private double[] weight;
	/**
	 * @uml.property  name="b"
	 */
	private double b;
	/**
	 * @uml.property  name="fun"
	 */
	private int fun;
	private static Random gen = new Random();
	
	/**
	 * getter for additional input
	 * @return  additional input
	 * @uml.property  name="b"
	 */
	public double getB() {
		return b;
	}

	/**
	 * setter for additional input
	 * @param b  additional input
	 * @uml.property  name="b"
	 */
	public void setB(double b) {
		this.b = b;
	}

	/**
	 * getter for weights
	 * @return  weights
	 * @uml.property  name="weight"
	 */
	public double[] getWeight() {
		return weight;
	}

	/**
	 * setter for weights
	 * @param weight  - weights of nueron
	 * @uml.property  name="weight"
	 */
	public void setWeight(double[] weight) {
		this.weight = weight;
	}

	/**
	 * constructor with random weights
	 * @param f - number of neuron function (1 - gaus, 2 - sigmoid, 3 - line)
	 * @param numberOfInputs - number of neuron inputs
	 */
	public Neuron(int f, int numberOfInputs) {
		fun = f;
		weight = new double[numberOfInputs];
		for(int i = 0; i < weight.length; ++i) {
			weight[i] = ((gen.nextDouble() * 1.0) - 0.5) * 1.3;
		}
		b = ((gen.nextDouble() * 1.0) - 0.5) * 1.3;
	}
	
	/**
	 * constructor with fixed weights
	 * @param f - number of neuron function (1 - gaus, 2 - sigmoid, 3 - line)
	 * @param numberOfInputs - number of neuron inputs
	 * @param value - value of weights
	 */
	public Neuron(int f, int numberOfInputs, double value) {
		fun = f;
		weight = new double[numberOfInputs];
		for(int i = 0; i < weight.length; ++i) {
			weight[i] = value;
		}
		b = value;
	}
	
	/**
	 * Run a neuron
	 * @param input - neuron input
	 * @return result of neuron work
	 * @throws Exception on wrong size of input array
	 */
	public double runNeuron(double[] input) throws Exception {
		if(input.length == weight.length) {
			double funInput = 0.0;
			for(int i = 0; i < weight.length; ++i) {
				funInput += input[i]*weight[i];
			}
			return applyFun(funInput + b);
		} else {
			throw new Exception("Wrong input size.");
		}
	}
	
	/**
	 * Apply a neuron function
	 * @param value - function argument
	 * @return result of function from argument
	 * @throws Exception at unknow function number
	 */
	private double applyFun(double value) throws Exception {
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
	
	/**
	 * line function
	 * @param value - function argument
	 * @return esult of function from argument
	 */
	private double lineFun(double value) {
		return value;
	}
	
	/**
	 * sigmoid function
	 * @param value - function argument
	 * @return result of function from argument
	 */
	private double sigmFun(double value) {
		return (1/(1 + java.lang.Math.pow(java.lang.Math.E, -value)));
	}
	
	/**
	 * gaus function with m = 0, d = 1
	 * @param value - function argument
	 * @return result of function from argument
	 */
	private double gausFun(double value) {
		double m = 0.0;
		double d = 1.0;
		double res = 0.0;
		res = 1/(d * java.lang.Math.pow(2 * java.lang.Math.PI, 0.5));
		res *= java.lang.Math.pow(java.lang.Math.E, - ( java.lang.Math.pow(value - m,2) / (2 * java.lang.Math.pow(d,2)) ) );
		return res;
	}
}
