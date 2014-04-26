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
	const int batchSize = input.BatchSize();

	const int ipEffectiveWidth = ipWidth + _padWidth * 2,
		      ipEffectiveHeight = ipHeight + _padHeight * 2;

	const int opWidth = (size_t) floor((ipEffectiveWidth - _windowSizeX + 1) / float(_strideX));
	const int opHeight = (size_t) floor((ipEffectiveHeight - _windowSizeY + 1) / float(_strideY));
	const int opDepth = _linearLayer.OutputSize();

	const auto Right = [this] (int val) { return val + _windowSizeX; };
	const auto Bottom = [this] (int val) { return val + _windowSizeY; };
	const auto IpRow = [ipWidth] (int y) { return y * ipWidth; };

	Params output(opWidth, opHeight, opDepth,
			CMatrix(opWidth * opHeight * opDepth, batchSize),
			Params::Packed);

	int xMax = ipWidth + _padWidth,
		yMax = ipHeight + _padHeight;

	int yConvoCurr = -_padHeight,
		xConvoCurr = -_padWidth;
	int xOpCurr = 0, yOpCurr = 0;

	// This object will store the partial convolution product of the current filter
	// application
	Vector convoPartialSum(batchSize);

	while (Bottom(yConvoCurr) <= yMax)
	{
		// Reset the x position
		xConvoCurr = -_padWidth;
		xOpCurr = 0;

		int yKernelStart = max(0, yConvoCurr);
		int yKernelEnd = min(Bottom(yConvoCurr), ipHeight);

		while (Right(xConvoCurr) <= xMax)
		{
			int xKernelStart = max(0, xConvoCurr);
			int xKernelEnd = min(Right(xConvoCurr), ipWidth);
			int xKernelSize = xKernelEnd - xKernelStart;

			// Iterate over all of the filters
			for (size_t filterIdx = 0; filterIdx < opDepth; ++filterIdx)
			{
				// Reset the partial sum
				convoPartialSum.setZero();

				for (int yKernel = yKernelStart; yKernel < yKernelEnd; ++yKernel)
				{
					// Get the parameters for the part of the current filter, based on the vertical part we are at
					auto filterBlock = weights.block(filterIdx, yKernel * _windowSizeX + xKernelStart, 1, xKernelSize);

					// Get the row index of the current row in the matrix
					int ipRowStart = IpRow(yKernel) + xKernelStart;

					auto inputBlock = input.Data.block(ipRowStart, 0, xKernelSize, batchSize);

					// The partial sum increases by the multiplication of the current row of the kernel
					// Being on the fringes is handled using the x and y kernel variables
					convoPartialSum.noalias() += filterBlock * inputBlock;
				}

				// Ok, now that we have fully applied the kernel, we can
				// write out the partial sum to the correct part of the output buffer
				// Sadly, all of this work only produces a single output pixel per batch image
				auto opBlock = output.Data.block(yOpCurr * opWidth * opDepth
													+ xOpCurr * opDepth
													+ filterIdx, 0, 1, batchSize);
				// Assign the block
				opBlock = convoPartialSum;
			}

			xConvoCurr += _strideX;
			xOpCurr++;
		}

		yConvoCurr += _strideY;
		yOpCurr++;
	}

	return move(output);
}

Params ConvoLayer::ComputePlanar(int threadIdx, const Params &unpaddedInput, bool isTraining)
{
	throw runtime_error("Planar input data is not currently supported.");
}

Params ConvoLayer::Backprop(int threadIdx, const Params &lastInput, const Params &lastOutput, const Params &outputErrors)
{
	return Params();
	/*MultiParams &linearInputs = _threadWindows[threadIdx];

	size_t opDepth = _linearLayer.OutputSize();

	size_t numOut = outputErrors.Data.size() / opDepth;

	// Get the output errors
	MultiParams linearOutputErrors(numOut, Params(1, 1, opDepth, Vector()));
	for (size_t i = 0, end = linearOutputErrors.size(); i < end; ++i)
	{
		linearOutputErrors[i].Data = outputErrors.Data.block(i * opDepth, 0, opDepth, 1);
	}

	MultiParams linearInputErrors = _linearLayer.BackpropMany(threadIdx, linearInputs, linearOutputErrors);

	// The input error is the windowed sum of the linear input errors
	Params paddedInputErrors = GetZeroPaddedInput(lastInput);

	RMap paddedMap(paddedInputErrors.Data.data(), paddedInputErrors.Height, paddedInputErrors.Width * paddedInputErrors.Depth);

	//Matrix wndMat(_windowSizeY, _windowSizeX * opDepth);
	size_t wndWidth = _windowSizeX * lastInput.Depth;
	size_t wndHeight = _windowSizeY;

	for (size_t ipY = 0, errIdx = 0; ipY < paddedInputErrors.Height - _windowSizeY + 1; ipY += _strideY)
	{
		for (size_t ipX = 0; 
				ipX < (paddedInputErrors.Width - _windowSizeX + 1) * lastInput.Depth; 
				ipX += _strideX * lastInput.Depth, ++errIdx)
		{
			Params &linearIpErr = linearInputErrors[errIdx];

			RMap mIpErr(linearIpErr.Data.data(), wndHeight, wndWidth);

			paddedMap.block(ipY, ipX, wndHeight, wndWidth) += mIpErr;
		}
	}

	if (_padMode == NoPadding)
		return move(paddedInputErrors);

	Params unpaddedInputErrors(lastInput, Vector(lastInput.size()));

	RMap mUpInput(unpaddedInputErrors.Data.data(), lastInput.Height, lastInput.Width * lastInput.Depth);

	mUpInput = paddedMap.block(_windowSizeY / 2, (_windowSizeX / 2) * lastInput.Depth,
		lastInput.Height, lastInput.Width * lastInput.Depth);

	return move(unpaddedInputErrors);*/
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


