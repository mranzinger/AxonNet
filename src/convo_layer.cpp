#include "convo_layer.h"

#include "util/enum_to_string.h"
#include "memset_util.h"


using namespace std;
using namespace axon::serialization;

ConvoLayer::ConvoLayer(string name, 
						size_t inputDepth, size_t outputDepth, 
						size_t windowSizeX, size_t windowSizeY, 
						size_t strideX, size_t strideY, 
						int padWidth, int padHeight)
	: LayerBase(move(name)), 
	  	_inputDepth(inputDepth),
		_linearLayer("", inputDepth * windowSizeX * windowSizeY, outputDepth),
		_windowSizeX(windowSizeX), _windowSizeY(windowSizeY), 
		_strideX(strideX), _strideY(strideY),
		_padWidth(padWidth), _padHeight(padHeight)
{
	PrepareForThreads(1);
}

ConvoLayer::ConvoLayer(std::string name,
						RMatrix linWeights, Vector linBias,
						size_t windowSizeX, size_t windowSizeY,
						size_t strideX, size_t strideY,
						int padWidth, int padHeight)
	: LayerBase(move(name)),
		_linearLayer("", move(linWeights), move(linBias)),
		_windowSizeX(windowSizeX), _windowSizeY(windowSizeY),
		_strideX(strideX), _strideY(strideY),
		_padWidth(padWidth), _padHeight(padHeight)
{
	PrepareForThreads(1);
}

Params ConvoLayer::Compute(int threadIdx, const Params &unpaddedInput, bool isTraining)
{
	if (_linearLayer.InputSize() != unpaddedInput.Depth * _windowSizeX * _windowSizeY)
	{
		assert(false);
		throw runtime_error("The underlying linear layer doesn't take the correct input dimensions.");
	}

	switch (unpaddedInput.Layout)
	{
	case Params::Packed:
		return ComputePacked(threadIdx, unpaddedInput, isTraining);
	case Params::Planar:
		return ComputePlanar(threadIdx, unpaddedInput, isTraining);
	default:
		throw runtime_error("Unsupported parameter layout.");
	}
}

Params ConvoLayer::ComputePacked(int threadIdx, const Params &input, bool isTraining)
{
	LinParams &threadPrms = _linearLayer.GetParams(threadIdx);
	const RMatrix &weights = threadPrms.Weights;
	const Vector &biases = threadPrms.Biases;

	const int ipWidth = input.Width;
	const int ipHeight = input.Height;
	const int ipDepth = input.Depth;
	const int batchSize = input.BatchSize();

	const int ipStride = ipWidth * ipDepth;

	const int ipEffectiveWidth = ipWidth + _padWidth * 2,
		      ipEffectiveHeight = ipHeight + _padHeight * 2;

	const int opWidth = (size_t) floor((ipEffectiveWidth - _windowSizeX) / float(_strideX)) + 1;
	const int opHeight = (size_t) floor((ipEffectiveHeight - _windowSizeY) / float(_strideY)) + 1;
	const int opDepth = _linearLayer.OutputSize();
	const int opStride = opWidth * opDepth;

	const auto Right = [this] (int val) { return val + _windowSizeX; };
	const auto Bottom = [this] (int val) { return val + _windowSizeY; };
	const auto IpRow = [ipWidth] (int y) { return y * ipWidth; };

	Params output(opWidth, opHeight, opDepth,
			CMatrix(opWidth * opHeight * opDepth, batchSize),
			Params::Packed);

	int xMax = ipWidth + _padWidth,
		yMax = ipHeight + _padHeight;

	// This object will store the partial convolution product of the current filter
	// application
	Vector convoPartialSum(opDepth);

	for (int imageIdx = 0; imageIdx < batchSize; ++imageIdx)
	{
		int yOpCurr = 0;
		int yConvoCurr = -_padHeight;

		// Lambda that returns a block of image data
		auto GetImageBlock = [this, &input, ipStride, ipDepth, imageIdx]
		                (int row, int col, int size) { return input.Data.block(row * ipStride + col * ipDepth, imageIdx, size * ipDepth, 1); };

		while (Bottom(yConvoCurr) <= yMax)
		{
			// Reset the x position
			int xConvoCurr = -_padWidth;
			int xOpCurr = 0;

			int yKernelStart = max(0, yConvoCurr);
			int yKernelEnd = min(Bottom(yConvoCurr), ipHeight);

			int skipTop = max(0, -yConvoCurr);

			while (Right(xConvoCurr) <= xMax)
			{
				int xKernelStart = max(0, xConvoCurr);
				int xKernelEnd = min(Right(xConvoCurr), ipWidth);
				int xKernelSize = xKernelEnd - xKernelStart;

				int skipLeft = max(0, -xConvoCurr); // This will be greater than 0 when xConvoCurr is < 0

				// Always start the partial sum as the biases
				convoPartialSum = biases;

				for (int filterRow = 0; (filterRow + yKernelStart) < yKernelEnd; ++filterRow)
				{
					int kernelColStart = (filterRow + skipTop) * _windowSizeX * ipDepth;
					kernelColStart += skipLeft * ipDepth; // Also, use skip left to index into the current row

					// Construct a block from the kernel filters of the given row
					auto filterBlock = weights.block(0, kernelColStart, opDepth, xKernelSize * ipDepth);

					auto imgBlock = GetImageBlock(filterRow + yKernelStart, xKernelStart, xKernelSize);

					convoPartialSum += filterBlock * imgBlock;
				}

				// Get the assignment block
				auto opBlock = output.Data.block(yOpCurr * opStride + xOpCurr * opDepth, imageIdx, opDepth, 1);

				// Assign the now complete sum to the output block
				opBlock = convoPartialSum;

				xConvoCurr += _strideX;
				xOpCurr++;
			}

			yConvoCurr += _strideY;
			yOpCurr++;
		}
	}

	return move(output);
}

Params ConvoLayer::ComputePlanar(int threadIdx, const Params &unpaddedInput, bool isTraining)
{
	throw runtime_error("Planar input data is not currently supported.");
}

Params ConvoLayer::Backprop(int threadIdx, const Params &lastInput, const Params &lastOutput, const Params &pOutputErrors)
{
	LinParams &prms = _linearLayer.GetParams(threadIdx);
	const RMatrix &weights = prms.Weights;
	const Vector &biases = prms.Biases;

	RMatrix transWeights = weights.transpose();
	CMatrix transLastInput = lastInput.Data.transpose();

	const int ipWidth = lastInput.Width;
	const int ipHeight = lastInput.Height;
	const int ipDepth = lastInput.Depth;
	const int batchSize = lastInput.BatchSize();

	const int ipStride = ipWidth * ipDepth;

	const int ipEffectiveWidth = ipWidth + _padWidth * 2,
		      ipEffectiveHeight = ipHeight + _padHeight * 2;

	const int opWidth = (size_t) floor((ipEffectiveWidth - _windowSizeX) / float(_strideX)) + 1;
	const int opHeight = (size_t) floor((ipEffectiveHeight - _windowSizeY) / float(_strideY)) + 1;
	const int opDepth = _linearLayer.OutputSize();
	const int opStride = opWidth * opDepth;

	const auto Right = [this] (int val) { return val + _windowSizeX; };
	const auto Bottom = [this] (int val) { return val + _windowSizeY; };
	const auto IpRow = [ipWidth] (int y) { return y * ipWidth; };

	int xMax = ipWidth + _padWidth,
		yMax = ipHeight + _padHeight;

	Params pInputErrors(lastInput,
				CMatrix::Zero(lastInput.Data.rows(), lastInput.Data.cols()));

	const CMatrix &outputErrors = pOutputErrors.Data;
	CMatrix &inputErrors = pInputErrors.Data;

	for (int imageIdx = 0; imageIdx < batchSize; ++imageIdx)
	{
		int yOpCurr = 0;
		int yConvoCurr = -_padHeight;

		while (Bottom(yConvoCurr) <= yMax)
		{
			int xOpCurr = 0;
			int xConvoCurr = -_padWidth;

			int yKernelStart = max(0, yConvoCurr);
			int yKernelEnd = min(Bottom(yConvoCurr), ipHeight);

			int skipTop = max(0, -yConvoCurr);

			while (Right(xConvoCurr) <= xMax)
			{
				int xKernelStart = max(0, xConvoCurr);
				int xKernelEnd = min(Right(xConvoCurr), ipWidth);
				int xKernelSize = xKernelEnd - xKernelStart;

				int skipLeft = max(0, -xConvoCurr);

				// Get the output error for the current application of the kernel
				auto opErrBlock = outputErrors.block(
									yOpCurr * opStride + xOpCurr * opDepth,
									imageIdx,
									opDepth,
									1);

				// Update the bias gradient
				prms.BiasGrad.noalias() += opErrBlock;

				for (int filterRow = 0; (filterRow + yKernelStart) < yKernelEnd; ++filterRow)
				{
					int kernelColStart = (filterRow + skipTop) * _windowSizeX * ipDepth;
					kernelColStart += skipLeft * ipDepth;

					// In transpose space, each filter occupies a column, so
					// kernelColStart maps to the rows
					auto kernelBlock = transWeights.block(kernelColStart,
														  0,
														  xKernelSize * ipDepth,
														  opDepth);

					int inBuffStart = (filterRow + yKernelStart) * ipStride
											+ xKernelStart * ipDepth;
					int inBuffSize = xKernelSize * ipDepth;

					// Indexing a 4D object using 2 dimensions is... ugly.
					auto ipErrBlock = inputErrors.block(
										inBuffStart,
										imageIdx,
										inBuffSize,
										1);

					ipErrBlock.noalias() += kernelBlock * opErrBlock;

					// In transpose space, each input image occupies a row
					auto ipBlock = transLastInput.block(
										imageIdx,
										inBuffStart,
										1,
										inBuffSize
									);
					auto gradWeightsBlock = prms.WeightsGrad.block(
												0, kernelColStart, opDepth, inBuffSize);

					gradWeightsBlock.noalias() += opErrBlock * ipBlock;
				}

				xConvoCurr += _strideX;
				++xOpCurr;
			}

			yConvoCurr += _strideY;
			++yOpCurr;
		}
	}

	return move(inputErrors);
}

void ConvoLayer::ApplyDeltas()
{
	_linearLayer.ApplyDeltas();
}

void ConvoLayer::ApplyDeltas(int threadIdx)
{
	_linearLayer.ApplyDeltas(threadIdx);
}

void ConvoLayer::PrepareForThreads(size_t num)
{
	_threadWindows.resize(num);

	_linearLayer.PrepareForThreads(num);
}

void ConvoLayer::SyncWithHost()
{
	_linearLayer.SyncWithHost();
}

void ConvoLayer::InitializeFromConfig(const LayerConfig::Ptr &config)
{
	LayerBase::InitializeFromConfig(config);

	auto conv = dynamic_pointer_cast<ConvoLayerConfig>(config);

	if (!conv)
		throw runtime_error("The specified config is not for a convolutional layer.");

	_linearLayer.InitializeFromConfig(conv->LinearConfig);
}

LayerConfig::Ptr ConvoLayer::GetConfig() const
{
	auto ret = make_shared<ConvoLayerConfig>();
	BuildConfig(*ret);
	return ret;
}

void ConvoLayer::BuildConfig(ConvoLayerConfig &config) const
{
	LayerBase::BuildConfig(config);

	config.LinearConfig = _linearLayer.GetConfig();
}

void BindStruct(const CStructBinder &binder, ConvoLayerConfig &config)
{
	BindStruct(binder, (LayerConfig&) config);

	binder("linearConfig", config.LinearConfig);
}

void BindStruct(const CStructBinder &binder, ConvoLayer &layer)
{
	BindStruct(binder, (LayerBase&) layer);

	binder("windowSizeX", layer._windowSizeX)
		  ("windowSizeY", layer._windowSizeY)
		  ("strideX", layer._strideX)
		  ("strideY", layer._strideY)
		  ("padWidth", layer._padWidth)
		  ("padHeight", layer._padHeight)
		  ("inputDepth", layer._inputDepth)
		  ;
}

AXON_SERIALIZE_DERIVED_TYPE(LayerConfig, ConvoLayerConfig, ConvoLayerConfig);

AXON_SERIALIZE_DERIVED_TYPE(ILayer, ConvoLayer, ConvoLayer);


