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
	: _batchSize(32)
    //: _batchSize(128)
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

	SetCost(config->Cost);
	_bestCorr = config->BestCorr;
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
	ret->BestCorr = _bestCorr;

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

	_learnRate = rate;
}

Params NeuralNet::Compute(const Params &input)
{
	return Compute(0, input, false);
}

Params NeuralNet::Compute(int threadIdx, const Params &input, bool isTraining)
{
	Params tmp(input);

	for (auto layer : _layers)
	{
		tmp = layer->Compute(threadIdx, tmp, isTraining);
	}

	return move(tmp);
}

Real NeuralNet::GetCost(const Params &pred, const Params &labels)
{
	return _cost->Compute(pred, labels);
}

BPStat NeuralNet::Backprop(int threadIdx, const Params &input, const Params &labels)
{
	MultiParams inputs(_layers.size() + 1);
	inputs[0] = input;

	for (size_t i = 0; i < _layers.size(); ++i)
	{
		inputs[i+1] = _layers[i]->Compute(threadIdx, inputs[i], true);
	}

	const Params &output = inputs.back();

	if (output.size() != labels.size())
		throw runtime_error("The output result set size doesn't match the label size.");

	Params opErr = _cost->ComputeGrad(output, labels);

	Real totalErr = GetCost(output, labels);

	Params corrCp(output);
	MaxBinarize(corrCp.Data);

#if _DEBUG
	MultiParams opErrs;
	opErrs.push_back(opErr);

	if (totalErr != 0)
	{
		for (int i = _layers.size() - 1; i >= 0; --i)
		{
			Params err = _layers[i]->Backprop(threadIdx, inputs[i], inputs[i + 1], opErrs.back());

			opErrs.push_back(move(err));
		}

		ApplyDeltas(threadIdx);
	}
#else
	if (totalErr != 0)
	{
		for (int i = _layers.size() - 1; i >= 0; --i)
		{
			opErr = _layers[i]->Backprop(threadIdx, inputs[i], inputs[i + 1], opErr);
		}

		ApplyDeltas(threadIdx);
	}
#endif

	return { totalErr, EqCount(corrCp.Data, labels.Data) };
}

struct ThreadTrainConfig
{
	int ThreadIdx = 0;
	event GoEvent;
	event DoneEvent;
	ITrainProvider *Provider = nullptr;
	size_t NumIters;
	Real BatchErr = 0.0f;
	Real NumCorrect = 0.0f;
	bool Kill = false;

	ThreadTrainConfig()
		: GoEvent(true), DoneEvent(true), NumIters(1) { }
};

void NeuralNet::Train(ITrainProvider &provider, size_t maxIters, size_t testFreq,
					  const std::string &chkRoot)
{
	static const int s_NumThreads = 1;
	static const int s_NumIters = 128 / _batchSize;

	PrepareThreads(s_NumThreads);

	ThreadTrainConfig configs[s_NumThreads];
	thread threads[s_NumThreads];

	for (size_t i = 0; i < s_NumThreads; ++i)
	{
		configs[i].ThreadIdx = i;
		configs[i].Provider = &provider;
		configs[i].NumIters = s_NumIters;

		threads[i] = thread(&NeuralNet::RunTrainThread, this, ref(configs[i]));
	}

	size_t epoch = 0;
	size_t iter = 0;

	for (size_t i = 0; i < maxIters; ++i)
	{
		size_t numThreads = epoch >= 10 ? s_NumThreads : 1;

		if (epoch == 10)
		{
			for (auto &layer : _layers)
				layer->SyncWithHost();
		}

		auto tStart = high_resolution_clock::now();

		for (size_t j = 0; j < numThreads; ++j)
			configs[j].GoEvent.set();

		// Wait for all of the threads to finish
		for (size_t j = 0; j < numThreads; ++j)
			configs[j].DoneEvent.wait();

		auto tEnd = high_resolution_clock::now();

		double timeSec = double(duration_cast<nanoseconds>(tEnd - tStart).count()) / 1000000000.0;

		Real err = accumulate(begin(configs), end(configs), 0.0,
			[](Real curr, const ThreadTrainConfig &cfg) { return curr + cfg.BatchErr; });
		Real corr = accumulate(begin(configs), end(configs), 0.0f,
			[](Real curr, const ThreadTrainConfig &cfg) { return curr + cfg.NumCorrect; });

		cout << setw(7) << i << " "
			 << setw(10) << left << (err / (numThreads * s_NumIters)) << " "
			 << setw(10) << left << (corr / (numThreads * s_NumIters * _batchSize)) << " "
			 << setw(10) << left << timeSec << "s"
			 << endl;

		iter += numThreads * s_NumIters * _batchSize;

		if (iter >= testFreq)
		{
			iter = 0;
			++epoch;
			Test(provider, chkRoot);
		}
	}

	for (ThreadTrainConfig &cfg : configs)
	{
		cfg.Kill = true;
	}

	for (thread &t : threads)
	{
		t.join();
	}
}



void NeuralNet::RunTrainThread(ThreadTrainConfig &config)
{
    random_device rd;
	mt19937_64 engine(rd());
	uniform_int_distribution<> dist(0, config.Provider->Size() - 1);

	vector<size_t> idxs;

	while (!config.Kill)
	{
		config.GoEvent.wait();

		config.BatchErr = 0.0f;
		config.NumCorrect = 0.0f;

		idxs.resize(_batchSize);

		Params vals, labels;

		for (size_t i = 0; i < config.NumIters; ++i)
		{
			for (size_t &idx : idxs)
				idx = dist(engine);

			config.Provider->Get(idxs, vals, labels);

			BPStat stat = Backprop(config.ThreadIdx, vals, labels);

			config.BatchErr += stat.Error;

			config.NumCorrect += stat.NumCorrect;
		}

		config.DoneEvent.set();
	}
}

void NeuralNet::ApplyDeltas(int threadIdx)
{
	for (auto layer : _layers)
	{
		layer->ApplyDeltas(threadIdx);
	}
}

void NeuralNet::Test(ITrainProvider &provider, const std::string &chkRoot)
{
	if (chkRoot.empty())
		throw runtime_error("Invalid checkpoint root directory.");

	if (!fs::exists(fs::path(chkRoot)))
		fs::create_directories(fs::path(chkRoot));

	cout << "TEST    " << flush;

	auto tStart = high_resolution_clock::now();

	Params input, labels;

	Real testErr = 0;
	Real numCorr = 0;

	vector<size_t> idxs;
	for (size_t i = 0; i < max(_batchSize, 256ul); ++i)
	    idxs.push_back(i);

	for (size_t i = 0, end = provider.TestSize(); i < end; i += idxs.size())
	{
		size_t batchSize = min(idxs.size(), end - i);

		// This is a no-op if batchSize is the size of the indexes,
		// otherwise it will truncate the indexes that are out of range
		idxs.resize(batchSize);

		provider.GetTest(idxs, input, labels);

		Params op = Compute(0, input, false);

		Real err = _cost->Compute(op, labels);

		testErr += err;

		MaxBinarize(op.Data);

		numCorr += EqCount(op.Data, labels.Data);

		transform(idxs.begin(), idxs.end(), idxs.begin(),
		        [&idxs] (size_t v) { return v + idxs.size(); });
	}

	auto tEnd = high_resolution_clock::now();

	double timeSec = double(duration_cast<nanoseconds>(tEnd - tStart).count()) / 1000000000.0;

	testErr /= provider.TestSize();
	numCorr /= provider.TestSize();

	if (numCorr > _bestCorr)
	{
		_bestCorr = numCorr;
		SaveCheckpoint(chkRoot);
	}

	cout << setw(10) << testErr << " "
		 << setw(10) << numCorr << " "
		 << setw(10) << timeSec << "s"
		 << setw(10) << _bestCorr << " "
		 << endl;
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
		  ("cost", config.Cost)
		  ("bestCorr", config.BestCorr);
}

void WriteStruct(const axon::serialization::CStructWriter &writer, const NeuralNet &net)
{
	writer("layers", net._layers)
		  ("cost", net._cost);
}
void ReadStruct(const axon::serialization::CStructReader &reader, NeuralNet &net)
{
	reader("layers", net._layers)
		  ("cost", net._cost);

	for (auto layer : net._layers)
		layer->SetNet(&net);
	net._cost->SetNet(&net);
}
