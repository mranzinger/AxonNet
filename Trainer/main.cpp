#include <iostream>

#include <serialization/master.h>

#include "neural_net.h"
#include "linear_layer.h"
#include "neuron_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "convo_layer.h"
#include "logloss_cost.h"
#include "maxpool_layer.h"
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
	
	// Layer 1
	net.Add<ConvoLayer>("C1",
						1, 6, // Input, Output Depth
						5, 5, // Window Size X, Y
						1, 1, // Stride X, Y,
						ConvoLayer::ZeroPad); // Output: 28x28x6
	net.Add<HardTanhNeuronLayer>("C1-NL"); // Non-linearity for this layer

	// Layer 2
	net.Add<MaxPoolLayer>("MP2",
						  2, 2); // Window Size X, Y
								 // Output: 14x14x6
	// Layer 3
	net.Add<ConvoLayer>("C3",
						6, 16,
						5, 5,
						1, 1,
						ConvoLayer::NoPadding); // Output: 10x10x16
	net.Add<HardTanhNeuronLayer>("C3-NL");

	// Layer 4
	net.Add<MaxPoolLayer>("MP4",
						  2, 2); // Output: 5x5x16

	// Layer 5
	net.Add<ConvoLayer>("C5",
						16, 120,
						5, 5,
						1, 1,
						ConvoLayer::NoPadding); // Output: 1x1x120
	net.Add<HardTanhNeuronLayer>("L5-NL");

	// Layer 6
	net.Add<LinearLayer>("L6",
						 120,
						 84);
	net.Add<HardTanhNeuronLayer>("L6-NL");

	// Layer 7 - Output Layer
	net.Add<LinearLayer>("L7",
						 84,
						 outputSize);

	net.Add<SoftmaxLayer>("soft");
	net.SetCost<LogLossCost>();

	// Fully Connected Network
	/*net.Add<LinearLayer>("l1", inputSize, 500);
	net.Add<HardTanhNeuronLayer>("r1");
	net.Add<DropoutLayer>("d1");
	net.Add<LinearLayer>("l3", 500, 300);
	net.Add<HardTanhNeuronLayer>("r2");
	net.Add<DropoutLayer>("d3");
	net.Add<LinearLayer>("l4", 300, outputSize);
	net.Add<SoftmaxLayer>("soft");
	net.SetCost<LogLossCost>();*/

	if (argc == 2)
	{
		net.Load(argv[1]);
	}

	net.SetLearningRate(0.01);

	net.Train(loader, 100000000, 50000, "test");
}