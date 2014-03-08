#include "neural_net.h"

#include <filesystem>
#include <random>
#include <iostream>

#include "sum_sq_cost.h"

using namespace std;
using namespace axon::serialization;
namespace fs = tr2::sys;

NeuralNet::NeuralNet()
{
	SetCost(make_shared<SumSqCost>());
}

void NeuralNet::AddLayer(ILayer::Ptr layer)
{
	_layers.push_back(move(layer));
}

void NeuralNet::SetCost(ICost::Ptr cost)
{
	assert(cost);
	_cost = move(cost);
}

void NeuralNet::Load(const NetworkConfig::Ptr &config)
{
	for (auto lcfg : config->Configs)
	{
		ILayer::Ptr layer = FindLayer(lcfg->Name);

		if (layer)
			layer->InitializeFromConfig(lcfg);
	}

	//SetCost(config->Cost);
}

void NeuralNet::Load(const std::string &chkFile)
{
	NetworkConfig::Ptr config;
	CAxonSerializer().DeserializeFromFile(chkFile, config);
	Load(config);
}

NetworkConfig::Ptr NeuralNet::GetCheckpoint() const
{
	NetworkConfig::Ptr ret(new NetworkConfig);

	for (auto layer : _layers)
	{
		ret->Configs.push_back(layer->GetConfig());
	}

	ret->Cost = _cost;

	return move(ret);
}

ILayer::Ptr NeuralNet::FindLayer(const string &name) const
{
	for (auto layer : _layers)
	{
		if (layer->GetLayerName() == name)
			return layer;
	}
	return ILayer::Ptr();
}

void NeuralNet::SetLearningRate(Real rate)
{
	for (auto layer : _layers)
	{
		layer->SetLearningRate(rate);
	}

	_learnRate = rate;
}

Vector NeuralNet::Compute(const Vector &input)
{
	return Compute(0, input, false);
}

Vector NeuralNet::Compute(int threadIdx, const Vector &input, bool isTraining)
{
	Vector tmp(input);

	for (auto layer : _layers)
	{
		tmp = layer->Compute(threadIdx, tmp, isTraining);
	}

	return tmp;
}

Real NeuralNet::GetCost(const Vector &pred, const Vector &labels)
{
	return _cost->Compute(pred, labels);
}

Real NeuralNet::Backprop(int threadIdx, const Vector &input, const Vector &labels)
{
	vector<Vector> inputs(_layers.size() + 1);
	inputs[0] = input;

	for (size_t i = 0; i < _layers.size(); ++i)
	{
		inputs[i+1] = _layers[i]->Compute(threadIdx, inputs[i], true);
	}

	if (inputs.back().size() != labels.size())
		throw runtime_error("The output result set size doesn't match the label size.");

	Vector opErr = _cost->ComputeGrad(inputs.back(), labels);

	Real totalErr = GetCost(inputs.back(), labels);

	for (int i = _layers.size() - 1; i >= 0; --i)
	{
		opErr = _layers[i]->Backprop(threadIdx, inputs[i], inputs[i + 1], opErr);
	}

	ApplyDeltas();

	return totalErr;
}

void NeuralNet::Train(ITrainProvider &provider, size_t maxIters, size_t testFreq,
					  const std::string &chkRoot)
{
	Real bestError = numeric_limits<Real>::max();

	//std::default_random_engine engine(4211);
	std::random_device engine;

	std::uniform_int_distribution<> dist(0, provider.Size() - 1);

	Vector vals, labels;

	Real batchErr = 0;

	for (size_t i = 0; i < maxIters; ++i)
	{
		provider.Get(max(0, min(dist(engine), (int)provider.Size())), vals, labels);

		batchErr += Square(Backprop(0, vals, labels));

		if (i != 0)
		{
			if ((i % 128) == 0)
			{
				cout << "Batch Error: " << batchErr << endl;

				batchErr = 0;
			}

			if ((i % testFreq) == 0)
			{
				Test(provider, chkRoot, bestError);
			}
		}
	}
}

void NeuralNet::ApplyDeltas()
{
	for (auto layer : _layers)
	{
		layer->ApplyDeltas();
	}
}

void NeuralNet::Test(ITrainProvider &provider, const std::string &chkRoot, Real &bestError)
{
	if (chkRoot.empty())
		throw runtime_error("Invalid checkpoint root directory.");

	if (!fs::exists(fs::path(chkRoot)))
		fs::create_directories(fs::path(chkRoot));

	cout << "Testing..." << endl;

	Vector input, labels;

	Real testErr = 0;
	size_t numRight = 0, numWrong = 0;
	for (size_t i = 0, end = provider.TestSize(); i < end; ++i)
	{
		provider.GetTest(i, input, labels);

		Vector op = Compute(0, input, false);

		Real err = _cost->Compute(op, labels);

		testErr += err;

		MaxBinarize(op);

		if (op == labels)
			++numRight;
		else
			++numWrong;
	}

	if (testErr < bestError)
	{
		bestError = testErr;
		SaveCheckpoint(chkRoot);
	}

	cout << "Finished Testing. Error: " << testErr << endl
		<< "Best: " << bestError << endl
		<< "Num Right: " << numRight << endl
		<< "Num Wrong: " << numWrong << endl;
}

void NeuralNet::SaveCheckpoint(const std::string &chkRoot)
{
	NetworkConfig::Ptr chk = GetCheckpoint();

	fs::path fsRoot(chkRoot);

	fs::path savePath = fsRoot / fs::path("best.chk");

	CAxonSerializer().SerializeToFile(savePath.string(), chk);
}

void BindStruct(const CStructBinder &binder, NetworkConfig &config)
{
	binder("layers", config.Configs)
		("cost", config.Cost);
}

void BindStruct(const CStructBinder &binder, NeuralNet &net)
{
	binder("layers", net._layers)
		  ("cost", net._cost);
}