#include "logloss_cost.h"

#include "neural_net.h"
#include "softmax_layer.h"

using namespace std;

static const Real s_epss = 0.000000001;

LogLossCost::LogLossCost()
    : LogLossCost("", "")
{
}

LogLossCost::LogLossCost(std::string inputName)
    : LogLossCost(move(inputName), "")
{
}

LogLossCost::LogLossCost(std::string inputName, std::string labelName)
    : SimpleCost(move(inputName), move(labelName)),
      _checked(false), _outputIsSoftmax(false)
{
}

CostMap LogLossCost::SCompute(const Params &preds, const Params &labels)
{
	if (_cuImpl)
		return _cuImpl->Compute(preds, labels);

	Real logLoss = 0.0f;

	const CMatrix &mPreds = preds.GetHostMatrix();
	const CMatrix &mLabels = labels.GetHostMatrix();

	for (int col = 0, cEnd = preds.Cols; col < cEnd; ++col)
	{
		auto vPred = mPreds.col(col);
		auto vLabel = mLabels.col(col);

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

		logLoss += cVal;
	}

	CMatrix binPreds = mPreds;
	MaxBinarize(binPreds);

	Real numCorr = EqCount(binPreds, mLabels);

	return CostMap{ { CostMap::PRIMARY_NAME, logLoss },
	                { string("correct"), numCorr } };
}

Params LogLossCost::SComputeGrad(const Params &pred, const Params &labels)
{
	EstablishContext();

	// Softmax combined with logloss cost simplifies extremely well
	// to y_i - 1*(i == class)
	// Although this isn't strictly necessary to do explicitly, it
	// helps with numerical stability and performance. It also allows us to
	// avoid the epsilon divide by 0 issue.
	//
	// Also, this works with mini-batch too!

	if (_cuImpl)
		return _cuImpl->ComputeGrad(pred, labels);

	const CMatrix &mPreds = pred.GetHostMatrix();
	const CMatrix &mLabels = labels.GetHostMatrix();

	if (_outputIsSoftmax)
		return Params(pred, new CMatrix((mPreds - mLabels) / pred.Cols));

	Params ret(pred, new CMatrix(pred.Rows, pred.Cols));

	CMatrix &mRet = ret.GetHostMatrix();

	float errFactor = 1.0f / pred.Cols;

	for (int col = 0, cEnd = pred.Cols; col < cEnd; ++col)
	{
		auto vPredCol = mPreds.col(col);
		auto vLabelCol = mLabels.col(col);

		auto safe = vPredCol.unaryExpr(
			[](Real val)
			{
				return min(max(val, s_epss), 1 - s_epss);
			});

		auto vRetCol = mRet.col(col);

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

	if (_cuImpl)
		_cuImpl->SetOpIsSoftmax(_outputIsSoftmax);
}

void LogLossCost::OnInitCudaDevice(int deviceId)
{
	_cuImpl.reset(new CuLoglossCost(deviceId));
}

void BindStruct(const aser::CStructBinder &binder, LogLossCost &cost)
{
    BindStruct(binder, (SimpleCost&)cost);
}

AXON_SERIALIZE_DERIVED_TYPE(ICost, LogLossCost, LogLossCost);


