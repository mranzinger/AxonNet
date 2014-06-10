#include "neural_net.h"


#include <random>
#include <iostream>
#include <numeric>
#include <thread>
#include <iomanip>
#include <chrono>

#include "sum_sq_cost.h"
#include "event.h"

#if _WIN32
#include <filesystem>
#include <ppl.h>
#include <concrt.h>

namespace fs = tr2::sys;
#else
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;
#endif

using namespace std;
using namespace axon::serialization;
using namespace std::chrono;


NeuralNet::NeuralNet()
{
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

	SetCost(make_shared<SumSqCost>());
}

void NeuralNet::AddLayer(ILayer::Ptr layer)
{
	layer->SetNet(this);

	_layers.push_back(move(layer));
}

void NeuralNet::SetCost(ICost::Ptr cost)
{
	assert(cost);

	cost->SetNet(this);

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

	_bestCost = config->BestCost;
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

	ret->BestCost = _bestCost;

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

int NeuralNet::GetLayerIndex(const std::string& name) const
{
	for (int i = 0; i < _layers.size(); ++i)
	{
		if (_layers[i]->GetLayerName() == name)
			return i;
	}
	return -1;
}

int NeuralNet::GetLayerIndex(const ILayer* layer) const
{
	for (int i = 0; i < _layers.size(); ++i)
	{
		if (_layers[i].get() == layer)
			return i;
	}
	return -1;
}

void NeuralNet::SetLearningRate(Real rate)
{
	for (auto layer : _layers)
	{
		layer->SetLearningRate(rate);
	}
}

void NeuralNet::Compute(ParamMap &input, bool isTraining)
{
	for (const ILayer::Ptr &layer : _layers)
	{
		layer->Compute(input, isTraining);
	}
}

CostMap NeuralNet::GetCost(const ParamMap &inputs)
{
	return _cost->Compute(inputs);
}

CostMap NeuralNet::Backprop(ParamMap &inputs)
{
    Compute(inputs, true);

    ParamMap errMap;

	_cost->ComputeGrad(inputs, errMap);

	CostMap err = GetCost(inputs);

	for (int i = _layers.size() - 1; i >= 0; --i)
	{
	    _layers[i]->Backprop(inputs, errMap);
	}

	ApplyGradient();

	return move(err);
}

void NeuralNet::Train(ITrainProvider &provider,
                      size_t batchSize,
                      size_t maxIters,
                      size_t testFreq,
					  const std::string &chkRoot)
{
    size_t testNum = 0;

    if (testFreq == 0)
        testFreq = provider.TrainSize() / batchSize;

    for (size_t i = 0; i < maxIters; ++i)
    {
        auto tStart = high_resolution_clock::now();

        ParamMap inputs;
        provider.GetTrain(inputs, batchSize);

        CostMap err = Backprop(inputs);

        auto tEnd = high_resolution_clock::now();

        double timeSec = duration_cast<nanoseconds>(tEnd - tStart).count()
                           / 1000000000.0;

        err /= batchSize;

        PrintStats(i, timeSec, err);

        if (i > 0 && (i % testFreq) == 0)
        {
            Test(provider, batchSize, chkRoot, testNum++);
        }
    }
}

void NeuralNet::ApplyGradient()
{
	for (const ILayer::Ptr &layer : _layers)
	{
		layer->ApplyGradient();
	}
}

void NeuralNet::Test(ITrainProvider &provider,
                     size_t batchSize,
                     const std::string &chkRoot,
                     size_t testNum)
{
	if (chkRoot.empty())
		throw runtime_error("Invalid checkpoint root directory.");

	if (!fs::exists(fs::path(chkRoot)))
		fs::create_directories(fs::path(chkRoot));

	cout << "TEST    " << flush;

	auto tStart = high_resolution_clock::now();

	ParamMap inputs;

	CostMap testCost;

	for (size_t i = 0, end = provider.TestSize(); i < end; i += batchSize)
	{
		provider.GetTest(inputs, i, batchSize);

		Compute(inputs, false);

		CostMap batchCost = GetCost(inputs);

		testCost += batchCost;
	}

	testCost /= provider.TestSize();

	auto tEnd = high_resolution_clock::now();

	double timeSec = double(duration_cast<nanoseconds>(tEnd - tStart).count()) / 1000000000.0;

	if (_cost->IsBetter(testCost, _bestCost))
	{
		_bestCost = testCost;
		SaveCheckpoint(chkRoot);
	}

	for (const pair<string, Real> &bestCost : _bestCost)
	{
	    testCost[bestCost.first + "-best"] = bestCost.second;
	}

	PrintStats(testNum, timeSec, testCost);
}

void NeuralNet::PrintStats(size_t iteration, double timeSec, const CostMap& costs)
{
    cout << setw(7) << iteration << " ";

    for (const pair<string, Real> &cost : costs)
    {
        cout << cost.first << " "
             << setw(10) << left << cost.second << " ";
    }

    cout << setw(10) << left << timeSec << "s" << endl;

}

void NeuralNet::SaveCheckpoint(const std::string &chkRoot)
{
	NetworkConfig::Ptr chk = GetCheckpoint();

	fs::path fsRoot(chkRoot);

	fs::path savePath = fsRoot / fs::path("best.chk");

	CAxonSerializer().SerializeToFile(savePath.string(), chk);
}

void NeuralNet::SetDevicePreference(const IDevicePreference::Ptr& pref)
{
	for (const ILayer::Ptr &layer : _layers)
		layer->SetDevicePreference(pref);
	_cost->SetDevicePreference(pref);
}

void BindStruct(const CStructBinder &binder, NetworkConfig &config)
{
	binder("layers", config.Configs)
		  ("bestCost", config.BestCost);
}

void WriteStruct(const aser::CStructWriter &writer, const NeuralNet &net)
{
	writer("layers", net._layers)
		  ("cost", net._cost);
}
void ReadStruct(const aser::CStructReader &reader, NeuralNet &net)
{
	IDevicePreference::Ptr pref;

	reader("layers", net._layers)
		  ("cost", net._cost)
		  ("device", pref);

	for (auto layer : net._layers)
		layer->SetNet(&net);
	net._cost->SetNet(&net);

	if (pref)
		net.SetDevicePreference(pref);
}


