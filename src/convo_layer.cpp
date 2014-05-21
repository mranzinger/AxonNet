#include "convo_layer.h"

#include "util/enum_to_string.h"
#include "memset_util.h"
#include "thread/parallel_for.h"

using namespace std;
using namespace axon::serialization;

//#define SINGLE_IMAGE

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
	//const RMatrix &weights = threadPrms.Weights;
	const CMatrix weights = threadPrms.Weights;
	const Vector &biases = threadPrms.Biases;

	const int ipWidth = input.Width;
	const int ipHeight = input.Height;
	const int ipDepth = input.Depth;
	const int batchSize = input.BatchSize();

	const int ipStride = ipWidth * ipDepth;

	const int ipEffectiveWidth = ipWidth + _padWidth * 2,
		      ipEffectiveHeight = ipHeight + _padHeight * 2;

	const int opWidth = (int) floor((ipEffectiveWidth - _windowSizeX) / float(_strideX)) + 1;
	const int opHeight = (int) floor((ipEffectiveHeight - _windowSizeY) / float(_strideY)) + 1;
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

#ifdef SINGLE_IMAGE
	int dfMiniBatchSize = 1;
#else
	int dfMiniBatchSize = (int)ceil(batchSize / float(s_threadPool.NumThreads()));
#endif
	//for (int imageIdx = 0; imageIdx < batchSize; imageIdx += dfMiniBatchSize)
	ParallelFor(s_threadPool, 0, batchSize, dfMiniBatchSize,
	        [&] (int imageIdx)
	{
	    int miniBatchSize = min(dfMiniBatchSize, batchSize - imageIdx);

	    // This object will store the partial convolution product of the current filter
	    // application
#ifdef SINGLE_IMAGE
	    Vector convoPartialSum(opDepth);
#else
	    CMatrix convoPartialSum(opDepth, miniBatchSize);
#endif


#ifdef SINGLE_IMAGE
	    UMapVector vecIpImage(const_cast<Real*>(input.Data.data()) + imageIdx * (ipStride * ipHeight),
	                       ipStride * ipHeight);
	    UMapVector vecOpImage(output.Data.data() + imageIdx * (opStride * opHeight),
	                       opStride * opHeight);
#endif

		int yOpCurr = 0;
		int yConvoCurr = -_padHeight;

		// Lambda that returns a block of image data
#ifdef SINGLE_IMAGE
		/*auto GetImageBlock = [this, &input, ipStride, ipDepth, imageIdx]
		                (int row, int col, int size) { return input.Data.block(row * ipStride + col * ipDepth, imageIdx, size * ipDepth, 1); };*/
		auto GetImageBlock = [this, &vecIpImage, ipStride, ipDepth, imageIdx]
                        (int row, int col, int size) { return vecIpImage.segment(row * ipStride + col * ipDepth, size * ipDepth); };
#else
		auto GetImageBlock = [this, &input, ipStride, ipDepth, imageIdx, miniBatchSize]
		                (int row, int col, int size) { return input.Data.block(row * ipStride + col * ipDepth, imageIdx, size * ipDepth, miniBatchSize); };
#endif

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
#ifdef SINGLE_IMAGE
				convoPartialSum = biases;
#else
				convoPartialSum.colwise() = biases;
#endif

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
#ifdef SINGLE_IMAGE
				/*auto opBlock = output.Data.block(yOpCurr * opStride + xOpCurr * opDepth, imageIdx, opDepth, 1);*/
				auto opBlock = vecOpImage.segment(yOpCurr * opStride + xOpCurr * opDepth, opDepth);
#else
				auto opBlock = output.Data.block(yOpCurr * opStride + xOpCurr * opDepth, imageIdx, opDepth, miniBatchSize);
#endif

				// Assign the now complete sum to the output block
				opBlock = convoPartialSum;

				xConvoCurr += _strideX;
				xOpCurr++;
			}

			yConvoCurr += _strideY;
			yOpCurr++;
		}
	});

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

	CMatrix transWeights = weights.transpose();

#ifdef SINGLE_IMAGE
	//RMatrix transLastInput = lastInput.Data.transpose();
#else
	CMatrix transLastInput = lastInput.Data.transpose();
#endif

	const int ipWidth = lastInput.Width;
	const int ipHeight = lastInput.Height;
	const int ipDepth = lastInput.Depth;
	const int batchSize = lastInput.BatchSize();

	const int ipStride = ipWidth * ipDepth;

	const int ipEffectiveWidth = ipWidth + _padWidth * 2,
		      ipEffectiveHeight = ipHeight + _padHeight * 2;

	//const int opWidth = (size_t) floor((ipEffectiveWidth - _windowSizeX) / float(_strideX)) + 1;
	//const int opHeight = (size_t) floor((ipEffectiveHeight - _windowSizeY) / float(_strideY)) + 1;
	const int opWidth = lastOutput.Width;
	const int opHeight = lastOutput.Height;
	const int opDepth = lastOutput.Depth;
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

	// Initialize the gradient matrices
	prms.BiasGrad.resize(prms.Biases.size());
	prms.BiasGrad.setZero();

	CMatrix cWeightsGrad(prms.Weights.rows(), prms.Weights.cols());
	cWeightsGrad.setZero();

#ifdef SINGLE_IMAGE
	int dfMiniBatchSize = 1;
#else
	int dfMiniBatchSize = (int)ceil(batchSize / float(s_threadPool.NumThreads()));
#endif

	//for (int imageIdx = 0; imageIdx < batchSize; imageIdx += miniBatchSize)
	ParallelFor(s_threadPool, 0, batchSize, dfMiniBatchSize,
	        [&] (int imageIdx)
	{
	    int miniBatchSize = min(dfMiniBatchSize, batchSize - imageIdx);

#ifdef SINGLE_IMAGE
	    UMapVector vecIpErrs(inputErrors.data() + (imageIdx * ipStride * ipHeight),
	                         ipStride * ipHeight);
	    UMapVector vecOpErrs(const_cast<Real*>(outputErrors.data()) + (imageIdx * opStride * opHeight),
	                         opStride * opHeight);
	    UMapRowVector vecLastInput(const_cast<Real*>(lastInput.Data.data()) + (imageIdx * ipStride * ipHeight),
	                         ipStride * ipHeight);
#endif

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
#ifdef SINGLE_IMAGE
				/*auto opErrBlock = outputErrors.block(
									yOpCurr * opStride + xOpCurr * opDepth,
									imageIdx,
									opDepth,
									1);*/
				auto opErrBlock = vecOpErrs.segment(
				                    yOpCurr * opStride + xOpCurr * opDepth,
				                    opDepth);
#else
				auto opErrBlock = outputErrors.block(
									yOpCurr * opStride + xOpCurr * opDepth,
									imageIdx,
									opDepth,
									miniBatchSize);
#endif

				// Update the bias gradient
				auto bGrad = opErrBlock.rowwise().sum();
				prms.BiasGrad += bGrad;

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
#ifdef SINGLE_IMAGE
					/*auto ipErrBlock = inputErrors.block(
										inBuffStart,
										imageIdx,
										inBuffSize,
										1);*/
					auto ipErrBlock = vecIpErrs.segment(
					                    inBuffStart,
					                    inBuffSize);
#else
					auto ipErrBlock = inputErrors.block(
										inBuffStart,
										imageIdx,
										inBuffSize,
										miniBatchSize);
#endif

					ipErrBlock.noalias() += kernelBlock * opErrBlock;

					// In transpose space, each input image occupies a row
#ifdef SINGLE_IMAGE
					/*auto ipBlock = transLastInput.block(
										imageIdx,
										inBuffStart,
										1,
										inBuffSize
									);*/
					auto ipBlock = vecLastInput.segment(
					                    inBuffStart,
					                    inBuffSize);
#else
					auto ipBlock = transLastInput.block(
										imageIdx,
										inBuffStart,
										miniBatchSize,
										inBuffSize
									);
#endif

					auto gradWeightsBlock = cWeightsGrad.block(
												0, kernelColStart, opDepth, inBuffSize);

					gradWeightsBlock.noalias() += opErrBlock * ipBlock;
				}

				xConvoCurr += _strideX;
				++xOpCurr;
			}

			yConvoCurr += _strideY;
			++yOpCurr;
		}
	});

	prms.WeightsGrad = cWeightsGrad;

	prms.LearningRate2 = 1.f / (opWidth * opHeight);

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

void WriteStruct(const CStructWriter &writer, const ConvoLayer &layer)
{
	WriteStruct(writer, (const LayerBase &)layer);

	writer("windowSizeX", layer._windowSizeX)
		  ("windowSizeY", layer._windowSizeY)
		  ("strideX", layer._strideX)
		  ("strideY", layer._strideY)
		  ("padWidth", layer._padWidth)
		  ("padHeight", layer._padHeight)
		  ("inputDepth", layer._inputDepth)
		  ("outputDepth", layer._linearLayer.OutputSize())
		  ;
}

void ReadStruct(const CStructReader &reader, ConvoLayer &layer)
{
	ReadStruct(reader, (LayerBase&)layer);

	size_t outputDepth;

	reader("windowSizeX", layer._windowSizeX)
		  ("windowSizeY", layer._windowSizeY)
		  ("strideX", layer._strideX)
		  ("strideY", layer._strideY)
		  ("padWidth", layer._padWidth)
		  ("padHeight", layer._padHeight)
		  ("inputDepth", layer._inputDepth)
		  ("outputDepth", outputDepth)
		  ;

	layer._linearLayer = LinearLayer(
							"",
							layer._windowSizeX * layer._windowSizeY * layer._inputDepth,
							outputDepth);
}

AXON_SERIALIZE_DERIVED_TYPE(LayerConfig, ConvoLayerConfig, ConvoLayerConfig);

AXON_SERIALIZE_DERIVED_TYPE(ILayer, ConvoLayer, ConvoLayer);


