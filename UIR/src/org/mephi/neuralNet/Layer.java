package org.mephi.neuralNet;

public class Layer {
	private Neuron[] neurons;

	public void setNeurons(Neuron[] neurons) {
		this.neurons = neurons;
	}

	public Neuron[] getNeurons() {
		return neurons;
	}
	
	public int getSize(){
		return neurons.length;
	}
	
	public Layer(int f, int numberOfInputs, int numberOfOutputs)
	{
		neurons = new Neuron[numberOfOutputs];
		for(int i = 0; i < neurons.length; ++i)
		{
			neurons[i] = new Neuron(f,numberOfInputs);
		}
	}
	
	public Layer(int f, int numberOfInputs, int numberOfOutputs, double weightVal)
	{
		neurons = new Neuron[numberOfOutputs];
		for(int i = 0; i < neurons.length; ++i)
		{
			neurons[i] = new Neuron(f,numberOfInputs, weightVal);
		}
	}
	
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
