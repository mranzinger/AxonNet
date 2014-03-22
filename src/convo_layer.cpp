#include "convo_layer.h"

#include "util/enum_to_string.h"

using namespace std;
using namespace axon::serialization;

ConvoLayer::ConvoLayer(string name, 
						size_t inputDepth, size_t outputDepth, 
						size_t windowSizeX, size_t windowSizeY, 
						size_t strideX, size_t strideY, 
						PaddingMode padMode)
	: LayerBase(move(name)), 
		_linearLayer("", inputDepth * windowSizeX * windowSizeY, outputDepth),
		_windowSizeX(windowSizeX), _windowSizeY(windowSizeY), 
		_strideX(strideX), _strideY(strideY), _padMode(padMode)
{
	PrepareForThreads(1);
}

ConvoLayer::ConvoLayer(std::string name,
						Matrix linWeights, Vector linBias,
						size_t windowSizeX, size_t windowSizeY,
						size_t strideX, size_t strideY,
						PaddingMode padMode)
	: LayerBase(move(name)),
		_linearLayer("", move(linWeights), move(linBias)),
		_windowSizeX(windowSizeX), _windowSizeY(windowSizeY),
		_strideX(strideX), _strideY(strideY), _padMode(padMode)
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

Params ConvoLayer::ComputePacked(int threadIdx, const Params &unpaddedInput, bool isTraining)
{
	const Params pInput = GetPaddedInput(unpaddedInput);

	const size_t ipWidth = pInput.Width;
	const size_t ipHeight = pInput.Height;

	const size_t opWidth = (size_t) ceil((pInput.Width - _windowSizeX + 1) / float(_strideX));
	const size_t opHeight = (size_t) ceil((pInput.Height - _windowSizeY + 1) / float(_strideY));
	const size_t opDepth = _linearLayer.OutputSize();

	Params output(opWidth, opHeight, opDepth, Vector(opWidth * opHeight * opDepth));

	Params window(_windowSizeX, _windowSizeY, pInput.Depth, Vector(_windowSizeX * _windowSizeY * pInput.Depth));

	const size_t inputStride = pInput.Width * pInput.Depth;
	const size_t outputStride = opWidth * opDepth;
	const size_t windowSize = window.Height * window.Width * window.Depth;
	const size_t windowStride = window.Width * window.Depth;

	MultiParams &threadPrms = _threadWindows[threadIdx];
	threadPrms.clear();
	threadPrms.reserve(opWidth * opHeight);

	for (size_t ipY = 0, opIdx = 0; ipY < ipHeight - _windowSizeY + 1; ipY += _strideY)
	{
		for (size_t ipX = 0;
			ipX < (ipWidth - _windowSizeX + 1) * pInput.Depth;
			ipX += (_strideX * pInput.Depth), opIdx += opDepth)
		{
			const Real *srcPtr = pInput.Data.data() + (ipY * inputStride) + ipX;
			Real *wndPtr = window.Data.data();
			for (size_t wndY = 0; wndY < _windowSizeY; ++wndY, srcPtr += inputStride, wndPtr += windowStride)
			{
				copy(srcPtr, srcPtr + windowStride, wndPtr);
			}

			// Convolve this window, and write it into the output buffer
			_linearLayer.Compute(threadIdx, window, output.Data.data() + opIdx);

			// If training, store the inputs
			if (isTraining)
				threadPrms.push_back(window);
		}
	}

	return move(output);
}

Params ConvoLayer::ComputePlanar(int threadIdx, const Params &unpaddedInput, bool isTraining)
{
	throw runtime_error("Planar input data is not currently supported.");
}

Params ConvoLayer::Backprop(int threadIdx, const Params &lastInput, const Params &lastOutput, const Params &outputErrors)
{
	MultiParams &linearInputs = _threadWindows[threadIdx];

	size_t opDepth = _linearLayer.OutputSize();

	size_t numOut = outputErrors.Data.size() / opDepth;

	// Get the output errors
	MultiParams linearOutputErrors(numOut, Params(1, numOut, 1, Vector()));
	for (size_t i = 0, end = linearOutputErrors.size(); i < end; ++i)
	{
		linearOutputErrors[i].Data = outputErrors.Data.block(i * opDepth, 0, opDepth, 1);
	}

	MultiParams linearInputErrors = _linearLayer.BackpropMany(threadIdx, linearInputs, linearOutputErrors);

	// The input error is the windowed sum of the linear input errors
	Params paddedInputErrors = GetZeroPaddedInput(lastInput);

	Map paddedMap(paddedInputErrors.Data.data(), paddedInputErrors.Height, paddedInputErrors.Width * paddedInputErrors.Depth);

	//Matrix wndMat(_windowSizeY, _windowSizeX * opDepth);
	size_t wndWidth = _windowSizeX * lastInput.Depth;
	size_t wndHeight = _windowSizeY;

	size_t windowSize = wndWidth * wndHeight;

	for (size_t ipY = 0, errIdx = 0; ipY < paddedInputErrors.Height - _windowSizeY + 1; ipY += _strideY)
	{
		for (size_t ipX = 0; 
				ipX < (paddedInputErrors.Width - _windowSizeX + 1) * lastInput.Depth; 
				ipX += _strideX * lastInput.Depth, ++errIdx)
		{
			Params &linearIpErr = linearInputErrors[errIdx];

			paddedMap.block(ipY, ipX, wndHeight, wndWidth) +=
				Map(linearIpErr.Data.data(), wndHeight, wndWidth);
		}
	}

	if (_padMode == NoPadding)
		return move(paddedInputErrors);

	Params unpaddedInputErrors(lastInput, Vector(lastInput.Height * lastInput.Width * lastInput.Depth));

	Map(unpaddedInputErrors.Data.data(),
		unpaddedInputErrors.Height,
		unpaddedInputErrors.Width * unpaddedInputErrors.Depth)
		=
		paddedMap.block(_windowSizeY / 2, (_windowSizeX / 2) * lastInput.Depth,
						lastInput.Height, lastInput.Width * lastInput.Depth);

	return move(unpaddedInputErrors);
}

Params ConvoLayer::GetPaddedInput(const Params &input) const
{
	if (_padMode == NoPadding)
		return input;

	size_t halfWindowSizeX = _windowSizeX / 2,
		   halfWindowSizeY = _windowSizeY / 2;

	Params ret(input.Width + _windowSizeX - 1, input.Height + _windowSizeY - 1, input.Depth, Vector());
	ret.Data = Vector::Zero(ret.size());

	Map mapIn(const_cast<Real*>(input.Data.data()), input.Height, input.Width * input.Depth);
	Map mapOut(ret.Data.data(), ret.Height, ret.Width * ret.Depth);

	mapOut.block(halfWindowSizeY, halfWindowSizeX * ret.Depth, mapIn.outerSize(), mapIn.innerSize()) = mapIn;

	return move(ret);
}

Params ConvoLayer::GetZeroPaddedInput(const Params &reference) const
{
	if (_padMode == NoPadding)
	{
		return Params(reference, Vector::Zero(reference.Data.size()));
	}
	else
	{
		return Params(reference, Vector::Zero(
			(reference.Height + _windowSizeY - 1) *
			(reference.Width + _windowSizeX - 1) *
			reference.Depth));
	}
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

ENUM_IO_MAP(ConvoLayer::PaddingMode)
	ENMAP(ConvoLayer::ZeroPad, "ZeroPad")
	ENMAP(ConvoLayer::NoPadding, "NoPadding");

void BindStruct(const CStructBinder &binder, ConvoLayer &layer)
{
	BindStruct(binder, (LayerBase&) layer);

	binder("windowSizeX", layer._windowSizeX)
		  ("windowSizeY", layer._windowSizeY)
		  ("strideX", layer._strideX)
		  ("strideY", layer._strideY)
		  ("padMode", layer._padMode);
}

AXON_SERIALIZE_DERIVED_TYPE(LayerConfig, ConvoLayerConfig, ConvoLayerConfig);

AXON_SERIALIZE_DERIVED_TYPE(ILayer, ConvoLayer, ConvoLayer);