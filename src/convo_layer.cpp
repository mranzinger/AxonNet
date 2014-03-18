#include "convo_layer.h"

using namespace std;

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

}

Params ConvoLayer::Compute(int threadIdx, const Params &input, bool isTraining)
{
	if (_linearLayer.InputSize() != input.Depth * _windowSizeX * _windowSizeY)
	{
		assert(false);
		throw runtime_error("The underlying linear layer doesn't take the correct input dimensions.");
	}

	Params pInput = GetPaddedInput(input);

	size_t opWidth = (pInput.Width - _windowSizeX + 1) / _strideX;
	size_t opHeight = (pInput.Height - _windowSizeY + 1) / _strideY;
	size_t opDepth = _linearLayer.OutputSize();

	Params output(opWidth, opHeight, opDepth, Vector(opWidth * opHeight * opDepth));

	Params window(_windowSizeX, _windowSizeY, input.Depth, Vector(_windowSizeX * _windowSizeY * input.Depth));

	size_t inputStride = input.Width * input.Depth;
	size_t outputStride = opWidth * opDepth;
	size_t windowSize = window.Height * window.Width * window.Depth;

	Matrix wndMat(window.Height, window.Width * window.Depth);

	for (size_t y = 0, opIdx = 0; y < opHeight; y += _strideY)
	{
		for (size_t x = 0; x < opWidth; x += _strideX, opIdx += opDepth)
		{
			wndMat = StrideMat(const_cast<Real*>(
				input.Data.data() + (y * inputStride) + (x * input.Depth)), 
				window.Height, window.Width * window.Depth,
				Eigen::OuterStride<>(inputStride));

			window.Data = MapVector(wndMat.data(), windowSize);

			Params convout = _linearLayer.Compute(threadIdx, window, isTraining);

			output.Data.block(opIdx, 0, opDepth, 1) = convout.Data;
		}
	}

	return move(output);
}

Params ConvoLayer::Backprop(int threadIdx, const Params &lastInput, const Params &lastOutput, const Params &outputErrors)
{
	throw runtime_error("Not implemented yet.");
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

void ConvoLayer::ApplyDeltas()
{
	_linearLayer.ApplyDeltas();
}

void ConvoLayer::ApplyDeltas(int threadIdx)
{
	_linearLayer.ApplyDeltas(threadIdx);
}