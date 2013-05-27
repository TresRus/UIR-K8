package org.mephi.neuralNet;

public interface Train {
	void trainNet(double[] data) throws Exception;
	void addTrainNet(double[] data) throws Exception;
}
