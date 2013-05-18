package org.mephi.neuralNet;

public class Layer {
	private Neuron[] neurons;

	/**
	 * get neurons of layer
	 * @return array of neurons
	 */
	public Neuron[] getNeurons() {
		return neurons;
	}
	
	/**
	 * get layer size
	 * @return number of neurons in layer
	 */
	public int getSize(){
		return neurons.length;
	}
	
	/**
	 * constructor with random neurons weights
	 * @param f - number of neuron function (1 - gaus, 2 - sigmoid, 3 - line)
	 * @param numberOfInputs - number of neurons inputs
	 * @param numberOfOutputs - number of neurons in layer
	 */
	public Layer(int f, int numberOfInputs, int numberOfOutputs)
	{
		neurons = new Neuron[numberOfOutputs];
		for(int i = 0; i < neurons.length; ++i)
		{
			neurons[i] = new Neuron(f,numberOfInputs);
		}
	}
	
	/**
	 * constructor with fixed neurons weights
	 * @param f - number of neuron function (1 - gaus, 2 - sigmoid, 3 - line)
	 * @param numberOfInputs - number of neurons inputs
	 * @param numberOfOutputs - number of neurons in layer
	 * @param weightVal - value of weights
	 */
	public Layer(int f, int numberOfInputs, int numberOfOutputs, double weightVal)
	{
		neurons = new Neuron[numberOfOutputs];
		for(int i = 0; i < neurons.length; ++i)
		{
			neurons[i] = new Neuron(f,numberOfInputs, weightVal);
		}
	}
	
	/**
	 * Run a layer
	 * @param input - layer input
	 * @return array of results of neurons work
	 * @throws Exception on wrong size of input array
	 */
	public double[] runLayer(double[] input) throws Exception
	{
		try{
			double[] res = new double[neurons.length];
			for(int i = 0; i < neurons.length; ++i)
			{
				res[i] = neurons[i].runNeuron(input);
			}
			return res;
		}
		catch(Exception e)
		{
			throw e;
		}	
	}

}
