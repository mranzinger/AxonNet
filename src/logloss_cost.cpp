#include "logloss_cost.h"

#include "neural_net.h"
#include "softmax_layer.h"

using namespace std;

static const Real s_epss = 0.0000001;

Real LogLossCost::Compute(const Params &preds, const Params &labels)
{
	// Pull the prediction away from 0 and 1
	auto safe = preds.Data.unaryExpr(
		[](Real val)
		{
			return min(max(val, s_epss), 1 - s_epss);
		});

	Real ret = -labels.Data.binaryExpr(safe,
		[](Real label, Real pred)
		{
			return label * log(pred) + (1 - label) * log(1 - pred);
		}).sum();

	return ret;
}

Params LogLossCost::ComputeGrad(const Params &pred, const Params &labels)
{
	EstablishContext();

	// Softmax combined with logloss cost simplifies extremely well
	// to y_i - 1*(i == class)
	// Although this isn't strictly necessary to do explicitly, it
	// helps with numerical stability and performance. It also allows us to
	// avoid the epsilon divide by 0 issue
	if (_outputIsSoftmax)
		return Params(pred, pred.Data - labels.Data);

	auto safe = pred.Data.unaryExpr(
		[](Real val)
		{
			return min(max(val, s_epss), 1 - s_epss);
		});

	Vector ret = labels.Data.binaryExpr(safe,
			[] (Real label, Real pred)
			{
				return ((1 - label) / (1 - pred)) - (label / pred);
			});

	return ret;
}

void LogLossCost::EstablishContext()
{
	if (_checked)
		return;
	_checked = true;

	if (!_net)
	{
		_outputIsSoftmax = false;
		return;
	}

	ILayer::Ptr opLayer = _net->GetLayer(_net->NumLayers() - 1);

	auto sl = dynamic_cast<SoftmaxLayer*>(opLayer.get());

	_outputIsSoftmax = sl != nullptr;
}

void BindStruct(const axon::serialization::CStructBinder &, LogLossCost&) { }

AXON_SERIALIZE_DERIVED_TYPE(ICost, LogLossCost, LogLossCost);


