#pragma once

#include "i_layer.h"


class NEURAL_NET_API LayerBase
	: public virtual ILayer
{
scope_protected:
	std::string _name;

	IDevicePreference::Ptr _devicePref;

	NeuralNet *_net;

scope_public:
	typedef std::shared_ptr<LayerBase> Ptr;

	LayerBase();
	explicit LayerBase(std::string name);

	virtual const std::string &GetLayerName() const override {
		return _name;
	}

	virtual void SetLearningRate(Real rate) override {}
	virtual void SetMomentum(Real rate) override {}
	virtual void SetWeightDecay(Real rate) override {}

	virtual void InitializeFromConfig(const LayerConfig::Ptr &config);
	virtual LayerConfig::Ptr GetConfig() const override;

	virtual void ApplyGradient() override { }

	virtual void SetNet(NeuralNet *net) override { _net = net; }

	virtual void SetDevicePreference(IDevicePreference::Ptr pref) override;

#ifdef _UNIT_TESTS_
	void UTBackprop(ParamMap &computeMap, ParamMap &inputErrorMap)
	{
		Compute(computeMap, true);

		Backprop(computeMap, inputErrorMap);
	}
#endif

    friend void BindStruct(const aser::CStructBinder &binder, LayerBase &layer);

scope_protected:
	void BuildConfig(LayerConfig &config) const;

	Params *GetData(ParamMap &pMap, const std::string &name, bool enforce = true) const;
	const Params *GetData(const ParamMap &pMap, const std::string &name, bool enforce = true) const;

	virtual void OnInitialized();
	virtual void OnInitCPUDevice() { }
	virtual void OnInitCudaDevice(int deviceId) { }
};
