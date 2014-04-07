#include "logloss_cost.h"

#include "neural_net.h"
#include "softmax_layer.h"

using namespace std;

static const Real s_epss = 0.0000001;

Real LogLossCost::Compute(const Params &preds, const Params &labels)
{
	Real ret = 0.0f;

	for (int col = 0, cEnd = preds.Data.cols(); col < cEnd; ++col)
	{
		auto vPred = preds.Data.col(col);
		auto vLabel = labels.Data.col(col);

		// Pull the prediction away from 0 and 1
		auto safe = vPred.unaryExpr(
			[](Real val)
			{
				return min(max(val, s_epss), 1 - s_epss);
			});

		Real cVal = -vLabel.binaryExpr(safe,
			[](Real label, Real pred)
			{
				return label * log(pred) + (1 - label) * log(1 - pred);
			}).sum();

		ret += cVal;
	}

	// Average the cost
	ret /= preds.Data.cols();

	return ret;
}

Params LogLossCost::ComputeGrad(const Params &pred, const Params &labels)
{
	EstablishContext();

	// Softmax combined with logloss cost simplifies extremely well
	// to y_i - 1*(i == class)
	// Although this isn't strictly necessary to do explicitly, it
	// helps with numerical stability and performance. It also allows us to
	// avoid the epsilon divide by 0 issue.
	//
	// Also, this works with mini-batch too!
	if (_outputIsSoftmax)
		return Params(pred, pred.Data - labels.Data);

	Params ret(pred, CMatrix(pred.Data.rows(), pred.Data.cols()));

	float errFactor = 1.0f / pred.Data.cols();

	for (int col = 0, cEnd = pred.Data.cols(); col < cEnd; ++col)
	{
		auto vPredCol = pred.Data.col(col);
		auto vLabelCol = labels.Data.col(col);

		auto safe = vPredCol.unaryExpr(
			[](Real val)
			{
				return min(max(val, s_epss), 1 - s_epss);
			});

		auto vRetCol = ret.Data.col(col);

		vRetCol = vLabelCol.binaryExpr(safe,
				[errFactor] (Real label, Real pred)
				{
					return (((1 - label) / (1 - pred)) - (label / pred)) * errFactor;
				});
	}

	return move(ret);
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


