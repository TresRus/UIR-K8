package org.mephi.neuralNet;

import java.util.ArrayList;
import java.util.List;
import org.mephi.addClasses.MyMath;

public class TrainBE implements Train {
	/**
	 * @uml.property  name="net"
	 * @uml.associationEnd  
	 */
	private NeuroNet net;
	
	public TrainBE(NeuroNet nnet) {
		net = nnet;
	}
	
	@Override
	public void trainNet(double[] data) throws Exception {
		double n = 0.0000001;
		int iterat = net.getwSize();
		
		double[] input;
		double[] test;
		int m = (data.length - net.getNumOfInputs() - net.getNumOfOutputs() + 1);
		
		for(int i = 0; i < iterat; ++i) {
			for(int j = 0; j < m; ++j) {
				input = MyMath.getSubArr(j,net.getNumOfInputs(),data);
				test = MyMath.getSubArr(j+net.getNumOfInputs(),net.getNumOfOutputs(),data);
				trainStep(input, test, n);
			}
			System.out.printf("BE: %f\n", net.countNetFit(data));
		}
	}

	@Override
	public void addTrainNet(double[] data) throws Exception {
		double n = 0.0000001;
		int iterat = net.getwSize()/20;
		
		double[] input;
		double[] test;
		int m = (data.length - net.getNumOfInputs() - net.getNumOfOutputs() + 1);
		
		for(int i = 0; i < iterat; ++i) {
			for(int j = 0; j < m; ++j) {
				input = MyMath.getSubArr(j,net.getNumOfInputs(),data);
				test = MyMath.getSubArr(j+net.getNumOfInputs(),net.getNumOfOutputs(),data);
				trainStep(input, test, n);
			}
			System.out.printf("BE: %f\n", net.countNetFit(data));
		}
	}

	private void trainStep(double[] input, double[] testOut, double n) throws Exception {
		if(testOut.length == net.getLayers()[net.getLayers().length-1].getSize()) {
			List<double[]> outs = new ArrayList<double[]>();
			List<double[]> deltas = new ArrayList<double[]>();
			
			double[] out;
			double[] res = net.normalize(input);
			for(Layer l : net.getLayers()) {
				out = new double[l.getSize()];
				res = l.runLayer(res);
				for(int i = 0; i < out.length; ++i) {
					out[i] = res[i];
				}
				outs.add(out);
			}
			
			double[] delta;
			double a;
			double[] test = net.normalize(testOut);
			
			delta = new double[net.getLayers()[net.getLayers().length-1].getSize()];
			for(int i = 0; i < delta.length; ++i) {
				a = outs.get(net.getLayers().length-1)[i];
				delta[i] = a*(1-a)*(test[i] - a);
			}
			deltas.add(delta);
			
			double[] prevDelta;
			for(int i = net.getLayers().length-2; i >= 0; --i) {
				prevDelta = delta;
				delta = new double[net.getLayers()[i].getSize()];
				double sum;
				for(int j = 0; j < net.getLayers()[i].getSize();++j) {
					sum = 0.0;
					for(int k = 0; k < net.getLayers()[i+1].getSize(); ++k) {
						sum += prevDelta[k] * net.getLayers()[i+1].getNeurons()[k].getWeight()[j];
					}
					a = outs.get(i)[j];
					delta[j] = a*(1-a)*sum;
				}
				deltas.add(delta);
			}
			
			for(int i = 0; i < net.getLayers().length; ++i) {
				countNewWeight(net.getLayers()[net.getLayers().length-1-i],deltas.get(i),outs.get(net.getLayers().length-1-i),n);
			}
			
		} else {
			throw new Exception("Wrong test array size");
		}
	}
	
	private void countNewWeight(Layer l, double[] delta, double[] out, double n) {
		double dw;
		double[] w;
		for(int i = 0; i < l.getSize(); ++i) {
			w = l.getNeurons()[i].getWeight();
			dw = n * delta[i] * out[i];
			for(int j = 0; j < w.length; ++j) {
				w[j] = w[j] + dw;
			}
			l.getNeurons()[i].setWeight(w);
		}
	}
	
}
