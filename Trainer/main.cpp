#include <iostream>

#include "neural_net.h"
#include "fc_layer.h"
#include "handwritten_loader.h"

using namespace std;

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
	net.AddLayer(make_shared<LogisticFCLayer>("l1", inputSize, 300));
	net.AddLayer(make_shared<LogisticFCLayer>("l2", 300, 100));
	net.AddLayer(make_shared<LogisticFCLayer>("l3", 100, 100));
	net.AddLayer(make_shared<LogisticFCLayer>("l4", 100, outputSize));

	net.SetLearningRate(0.01);

	net.Train(loader, 100000000, 50000, "test");
}