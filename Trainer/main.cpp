#include <iostream>

#include <serialization/master.h>

#include "neural_net.h"
#include "linear_layer.h"
#include "neuron_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "convo_layer.h"
#include "logloss_cost.h"
#include "handwritten_loader.h"

using namespace std;
using namespace axon::serialization;

int main(int argc, char *argv [])
{
	string root = "C:\\Users\\Mike\\Documents\\Neural Net\\";

	HandwrittenLoader loader(root + "train-images.idx3-ubyte",
							 root + "train-labels.idx1-ubyte",
							 root + "t10k-images.idx3-ubyte",
							 root + "t10k-labels.idx1-ubyte");

	Params tmpInputs, tmpLabels;
	loader.Get(0, tmpInputs, tmpLabels);

	//ConvoLayer tmpConvo("Convo", 1, 3, 3, 3, 1, 1, ConvoLayer::ZeroPad);

	//Params convout = tmpConvo.Compute(0, tmpInputs, false);

	size_t inputSize = tmpInputs.size();
	size_t outputSize = tmpLabels.size();

	NeuralNet net;

	// Convolutional Network
	//net.Add<ConvoLayer>("c1",
	//					1, 5, // Input Depth, Output Depth
	//					7, 7, // Window Size X, Window Size Y
	//					4, 4, // Stride X, Stride Y
	//					ConvoLayer::ZeroPad);
	//net.Add<RectifierNeuronLayer>("r1");
	//net.Add<DropoutLayer>("d1");
	////net.Add<ConvoLayer>("c2",
	////					5, 12,
	////					7, 7,
	////					4, 4,
	////					ConvoLayer::ZeroPad); // Output Size: 28 / 4 = 7
	////net.Add<RectifierNeuronLayer>("r2");
	////net.Add<DropoutLayer>("d2");
	////net.Add<LinearLayer>("l3", 7 * 7 * 12, 100);
	//net.Add<LinearLayer>("l3", 7 * 7 * 5, 100);
	//net.Add<HardTanhNeuronLayer>("h3");
	//net.Add<DropoutLayer>("d3");
	//net.Add<LinearLayer>("l4", 100, outputSize);
	//net.Add<SoftmaxLayer>("soft");
	//net.SetCost<LogLossCost>();

	// Fully Connected Network
	net.Add<LinearLayer>("l1", inputSize, 500);
	net.Add<HardTanhNeuronLayer>("r1");
	net.Add<DropoutLayer>("d1");
	net.Add<LinearLayer>("l3", 500, 300);
	net.Add<HardTanhNeuronLayer>("r2");
	net.Add<DropoutLayer>("d3");
	net.Add<LinearLayer>("l4", 300, outputSize);
	net.Add<SoftmaxLayer>("soft");
	net.SetCost<LogLossCost>();

	if (argc == 2)
	{
		net.Load(argv[1]);
	}

	net.SetLearningRate(0.00001);

	net.Train(loader, 100000000, 50000, "test");
}