#pragma once

#include <vector>

#include "i_layer.h"
#include "i_train_provider.h"

class NEURAL_NET_API NetworkConfig
{
public:
	typedef std::shared_ptr<NetworkConfig> Ptr;

	std::vector<LayerConfig::Ptr> Configs;

	friend void BindStruct(const axon::serialization::CStructBinder &binder, NetworkConfig &config);
};

class NEURAL_NET_API NeuralNet
{
private:
	std::vector<ILayer::Ptr> _layers;
	Real _learnRate = 1.0;

public:
	void AddLayer(ILayer::Ptr layer);
	void Load(const NetworkConfig::Ptr &config);
	void Load(const std::string &chkFile);

	ILayer::Ptr FindLayer(const std::string &name) const;

	NetworkConfig::Ptr GetCheckpoint() const;

	void SetLearningRate(Real rate);

	Vector Compute(const Vector &input);
	Vector Compute(int threadIdx, const Vector &input, bool isTraining);

	Real Backprop(int threadIdx, const Vector &input, const Vector &labels);

	void Train(ITrainProvider &provider, size_t maxIters, size_t testFreq,
		       const std::string &chkRoot);

	friend void BindStruct(const axon::serialization::CStructBinder &binder, NeuralNet &config);

private:
	void Test(ITrainProvider &provider, const std::string &chkRoot, Real &bestError);
	void SaveCheckpoint(const std::string &chkRoot);
};
 