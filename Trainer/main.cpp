#include <iostream>

#include <serialization/master.h>

#include "neural_net.h"
#include "linear_layer.h"
#include "neuron_layer.h"
#include "handwritten_loader.h"

using namespace std;
using namespace axon::serialization;

int main(int argc, char *argv [])
{
	string root = "D:\\Users\\Mike\\Documents\\Neural Net\\";

	HandwrittenLoader loader(root + "train-images.idx3-ubyte",
							 root + "train-labels.idx1-ubyte",
							 root + "t10k-images.idx3-ubyte",
							 root + "t10k-labels.idx1-ubyte");

	Vector tmpInputs, tmpLabels;
	loader.Get(0, tmpInputs, tmpLabels);

	size_t inputSize = tmpInputs.size();
	size_t outputSize = tmpLabels.size();

	NeuralNet net;
	net.AddLayer(make_shared<LinearLayer>("l1", inputSize, 300));
	net.AddLayer(make_shared<LogisticNeuronLayer>("n1"));
	net.AddLayer(make_shared<LinearLayer>("l2", 300, 100));
	net.AddLayer(make_shared<LogisticNeuronLayer>("n2"));
	net.AddLayer(make_shared<LinearLayer>("l3", 100, 100));
	net.AddLayer(make_shared<LogisticNeuronLayer>("n3"));
	net.AddLayer(make_shared<LinearLayer>("l4", 100, outputSize));

	if (argc == 2)
	{
		net.Load(argv[1]);
	}

	net.SetLearningRate(0.0001);

	net.Train(loader, 100000000, 50000, "test");
}