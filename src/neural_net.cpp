#include "neural_net.h"

#include <filesystem>
#include <random>
#include <iostream>
#include <ppl.h>
#include <numeric>

#include "sum_sq_cost.h"

using namespace std;
using namespace axon::serialization;
namespace fs = tr2::sys;

NeuralNet::NeuralNet()
{
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

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

	if (totalErr != 0)
	{
		for (int i = _layers.size() - 1; i >= 0; --i)
		{
			opErr = _layers[i]->Backprop(threadIdx, inputs[i], inputs[i + 1], opErr);
		}

		ApplyDeltas(threadIdx);
	}

	return totalErr;
}

template<typename IdxType, typename Fn>
void fast_for(IdxType begin, IdxType end, Fn fn)
{
	Concurrency::parallel_for(begin, end, fn);
	//for (; begin != end; ++begin)
	//	fn(begin);
}

void NeuralNet::Train(ITrainProvider &provider, size_t maxIters, size_t testFreq,
					  const std::string &chkRoot)
{
	static const int s_NumThreads = 8;

	PrepareThreads(s_NumThreads);
	maxIters /= s_NumThreads;

	Real bestError = numeric_limits<Real>::max();

	std::random_device engines[s_NumThreads];
	std::uniform_int_distribution<> dists[s_NumThreads];
	for (auto &dist : dists)
		dist = uniform_int_distribution<>(0, provider.Size() - 1);

	// This is WAY not thread safe
	Real batchErrs[s_NumThreads] = { 0 };

	size_t iter = 0;
	size_t epoch = 0;
	for (size_t i = 0; i < maxIters; ++i)
	{
		int numThreads = epoch < 2 ? 1 : s_NumThreads;
		int workPerThread = 128 / numThreads;

		fast_for(0, numThreads,
			[&, this](int threadIdx)
			{
				auto &dist = dists[threadIdx];
				auto &engine = engines[threadIdx];
				Real &batchErr = batchErrs[threadIdx];
				Vector vals, labels;

				for (int p = 0; p < workPerThread; ++p)
				{
					provider.Get(max(0, min(dist(engine), (int) provider.Size())), vals, labels);

					batchErr += Backprop(threadIdx, vals, labels);
				}
			});

		Real batchErr = accumulate(batchErrs, batchErrs + s_NumThreads, 0.0);

		cout << "Batch Error: " << batchErr << endl;
		
		memset(batchErrs, 0, sizeof(batchErrs));

		iter += 128;

		if (iter >= testFreq)
		{
			iter = 0;
			++epoch;
			Test(provider, chkRoot, bestError);
		}
	}
}

void NeuralNet::ApplyDeltas(int threadIdx)
{
	for (auto layer : _layers)
	{
		layer->ApplyDeltas(threadIdx);
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

void NeuralNet::PrepareThreads(int numThreads)
{
	for (const auto &layer : _layers)
		layer->PrepareForThreads(numThreads);
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