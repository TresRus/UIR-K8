package org.mephi.predict;

import org.mephi.neuralNet.*;

public class Predict {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try{
			Neuron n = new Neuron(1,2,0.5);
			double[] in = new double[] {0,0};
			System.out.println(n.RunNeuron(in));
		}
		catch(Exception e)
		{
			System.out.println(e.getMessage());
		}
	}

}
