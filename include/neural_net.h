#pragma once

#include <xmmintrin.h>

#include <vector>

#include "i_layer.h"
#include "i_cost.h"
#include "i_train_provider.h"

class NEURAL_NET_API NetworkConfig
{
public:
	typedef std::shared_ptr<NetworkConfig> Ptr;

	std::vector<LayerConfig::Ptr> Configs;
	ICost::Ptr Cost;
	CostMap BestCost;

	friend void BindStruct(const aser::CStructBinder &binder, NetworkConfig &config);
};

struct ThreadTrainConfig;

class NEURAL_NET_API NeuralNet
{
private:
	std::vector<ILayer::Ptr> _layers;
	ICost::Ptr _cost;
	CostMap _bestCost;

public:
	NeuralNet();

	void AddLayer(ILayer::Ptr layer);
	void SetCost(ICost::Ptr cost);

	template<typename LayerType, typename ...Args>
	void Add(Args &&...args)
	{
		AddLayer(std::make_shared<LayerType>(std::forward<Args>(args)...));
	}

	template<typename CostType, typename ...Args>
	void SetCost(Args &&...args)
	{
		SetCost(std::make_shared<CostType>(std::forward<Args>(args)...));
	}

	void Load(const NetworkConfig::Ptr &config);
	void Load(const std::string &chkFile);

	ICost::Ptr GetCostFn() const { return _cost; }

	ILayer::Ptr GetLayer(size_t index) const { return _layers[index]; }
	ILayer::Ptr FindLayer(const std::string &name) const;
	int GetLayerIndex(const std::string &name) const;
	int GetLayerIndex(const ILayer *layer) const;

	size_t NumLayers() const { return _layers.size(); }

	NetworkConfig::Ptr GetCheckpoint() const;

	void SetLearningRate(Real rate);

	CostMap GetCost(const ParamMap &inputs);

	void Compute(ParamMap &inputs, bool isTraining = false);

	CostMap Backprop(ParamMap &inputs);

	void Train(ITrainProvider &provider,
	           size_t batchSize,
	           size_t maxIters,
	           size_t testFreq,
		       const std::string &chkRoot);

	friend void WriteStruct(const aser::CStructWriter &writer, const NeuralNet &net);
	friend void ReadStruct(const aser::CStructReader &reader, NeuralNet &net);

private:
	void ApplyGradient();
	void Test(ITrainProvider &provider, size_t batchSize,
	          const std::string &chkRoot, size_t testNum);
	void SaveCheckpoint(const std::string &chkRoot);
	void PrintStats(size_t iteration, double timeSec, const CostMap &cost);
};
 
