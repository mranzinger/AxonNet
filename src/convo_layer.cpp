#include "convo_layer.h"

#include "util/enum_to_string.h"
#include "memset_util.h"
#include "thread/parallel_for.h"

using namespace std;
using namespace axon::serialization;

ConvoLayer::ConvoLayer(string name, 
						size_t inputDepth, size_t outputDepth, 
						size_t windowSizeX, size_t windowSizeY, 
						size_t strideX, size_t strideY, 
						int padWidth, int padHeight)
	: SingleInputLayer(move(name)),
	  WeightLayer(inputDepth * windowSizeX * windowSizeY, outputDepth),
		_windowSizeX(windowSizeX), _windowSizeY(windowSizeY), 
		_strideX(strideX), _strideY(strideY),
		_padWidth(padWidth), _padHeight(padHeight)
{
}

ConvoLayer::ConvoLayer(std::string name,
						CMatrix linWeights, Vector linBias,
						size_t windowSizeX, size_t windowSizeY,
						size_t strideX, size_t strideY,
						int padWidth, int padHeight)
	: SingleInputLayer(move(name)),
	  WeightLayer(move(linWeights), move(linBias)),
		_windowSizeX(windowSizeX), _windowSizeY(windowSizeY),
		_strideX(strideX), _strideY(strideY),
		_padWidth(padWidth), _padHeight(padHeight)
{
}

void ConvoLayer::SetLearningRate(Real rate)
{
    WeightLayer::SetLearningRate(rate);
}

void ConvoLayer::SetMomentum(Real rate)
{
    WeightLayer::SetMomentum(rate);
}

void ConvoLayer::SetWeightDecay(Real rate)
{
    WeightLayer::SetWeightDecay(rate);
}

Params ConvoLayer::SCompute(const Params &input, bool isTraining)
{
	if (InputSize() != input.Depth * _windowSizeX * _windowSizeY)
	{
		assert(false);
		throw runtime_error("The underlying linear layer doesn't take the correct input dimensions.");
	}

	const CMatrix weights = _weights.Weights;
	const Vector &biases = _weights.Biases;

	const int ipWidth = input.Width;
	const int ipHeight = input.Height;
	const int ipDepth = input.Depth;
	const int batchSize = input.Cols;

	const int ipStride = ipWidth * ipDepth;

	const int ipEffectiveWidth = ipWidth + _padWidth * 2,
		      ipEffectiveHeight = ipHeight + _padHeight * 2;

	const int opWidth = (int) floor((ipEffectiveWidth - _windowSizeX) / float(_strideX)) + 1;
	const int opHeight = (int) floor((ipEffectiveHeight - _windowSizeY) / float(_strideY)) + 1;
	const int opDepth = OutputSize();
	const int opStride = opWidth * opDepth;

	const auto Right = [this] (int val) { return val + _windowSizeX; };
	const auto Bottom = [this] (int val) { return val + _windowSizeY; };
	const auto IpRow = [ipWidth] (int y) { return y * ipWidth; };

	Params output(opWidth, opHeight, opDepth,
			new CMatrix(opWidth * opHeight * opDepth, batchSize));

	int xMax = ipWidth + _padWidth,
		yMax = ipHeight + _padHeight;

	const int dfMiniBatchSize = 1;

	const CMatrix &mInput = input.GetHostMatrix();
	CMatrix &mOutput = output.GetHostMatrix();

	FastFor(GetThreadPool(), 0, batchSize, dfMiniBatchSize,
	        [&] (int imageIdx)
	{
	    int miniBatchSize = min(dfMiniBatchSize, batchSize - imageIdx);

	    // This object will store the partial convolution product of the current filter
	    // application
	    Vector convoPartialSum(opDepth);

	    UMapVector vecIpImage(const_cast<Real*>(mInput.data()) + imageIdx * (ipStride * ipHeight),
	                       ipStride * ipHeight);
	    UMapVector vecOpImage(mOutput.data() + imageIdx * (opStride * opHeight),
	                       opStride * opHeight);

		int yOpCurr = 0;
		int yConvoCurr = -_padHeight;

		// Lambda that returns a block of image data
		auto GetImageBlock = [this, &vecIpImage, ipStride, ipDepth, imageIdx]
                        (int row, int col, int size) { return vecIpImage.segment(row * ipStride + col * ipDepth, size * ipDepth); };

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
				auto opBlock = vecOpImage.segment(yOpCurr * opStride + xOpCurr * opDepth, opDepth);

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

Params ConvoLayer::SBackprop(const Params &lastInput, const Params &lastOutput, const Params &pOutputErrors)
{
	const RMatrix &weights = _weights.Weights;
	const Vector &biases = _weights.Biases;

	CMatrix transWeights = weights.transpose();

	const int ipWidth = lastInput.Width;
	const int ipHeight = lastInput.Height;
	const int ipDepth = lastInput.Depth;
	const int batchSize = lastInput.Cols;

	const int ipStride = ipWidth * ipDepth;

	const int ipEffectiveWidth = ipWidth + _padWidth * 2,
		      ipEffectiveHeight = ipHeight + _padHeight * 2;

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
				new CMatrix(CMatrix::Zero(lastInput.Rows, lastInput.Cols)));

	const CMatrix &mOutputErrors = pOutputErrors.GetHostMatrix();
	const CMatrix &mLastInput = lastInput.GetHostMatrix();
	CMatrix &mInputErrors = pInputErrors.GetHostMatrix();

	// Initialize the gradient matrices
	_weights.BiasGrad.resize(_weights.Biases.size());
	_weights.BiasGrad.setZero();

	CMatrix cWeightsGrad = CMatrix::Zero(weights.rows(), weights.cols());

	FastFor(GetThreadPool(), 0, batchSize, 1,
	        [&, this] (int imageIdx)
	{
	    Vector threadBiasGrad = Vector::Zero(_weights.BiasGrad.size());
	    CMatrix threadWeightsGrad = CMatrix::Zero(cWeightsGrad.rows(), cWeightsGrad.cols());

        UMapVector vecIpErrs(mInputErrors.data() + (imageIdx * ipStride * ipHeight),
                             ipStride * ipHeight);
        UMapVector vecOpErrs(const_cast<Real*>(mOutputErrors.data()) + (imageIdx * opStride * opHeight),
                             opStride * opHeight);
        UMapRowVector vecLastInput(const_cast<Real*>(mLastInput.data()) + (imageIdx * ipStride * ipHeight),
                             ipStride * ipHeight);

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
                auto opErrBlock = vecOpErrs.segment(
                                    yOpCurr * opStride + xOpCurr * opDepth,
                                    opDepth);

                // Update the bias gradient
                threadBiasGrad.noalias() += opErrBlock;

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
                    auto ipErrBlock = vecIpErrs.segment(
                                        inBuffStart,
                                        inBuffSize);

                    ipErrBlock.noalias() += kernelBlock * opErrBlock;

                    // In transpose space, each input image occupies a row
                    auto ipBlock = vecLastInput.segment(
                                        inBuffStart,
                                        inBuffSize);

                    auto gradWeightsBlock = threadWeightsGrad.block(
                                                0, kernelColStart, opDepth, inBuffSize);

                    gradWeightsBlock.noalias() += opErrBlock * ipBlock;
                } // End of Kernel Loop

                xConvoCurr += _strideX;
                ++xOpCurr;
            } // End of X loop

            yConvoCurr += _strideY;
            ++yOpCurr;
        } // End of Y loop

	    // Now we need to integrate all of the threaded gradients into the main gradients.
	    // Get a lock to the accumulator
	    lock_guard<mutex> lock(_bpLock);

	    _weights.BiasGrad += threadBiasGrad;
	    cWeightsGrad += threadWeightsGrad;
	});

	_weights.WeightsGrad = cWeightsGrad;

	_weights.DynamicLearningRate = 1.f / (opWidth * opHeight);

	return move(pInputErrors);
}

void ConvoLayer::InitializeFromConfig(const LayerConfig::Ptr &config)
{
	SingleInputLayer::InitializeFromConfig(config);
	WeightLayer::InitializeFromConfig(config);
}

LayerConfig::Ptr ConvoLayer::GetConfig() const
{
	auto cfg = WeightLayer::GetConfig();

	SingleInputLayer::BuildConfig(*cfg);

	return cfg;
}

void ConvoLayer::ApplyGradient()
{
    SingleInputLayer::ApplyGradient();
    WeightLayer::ApplyGradient();
}

size_t ConvoLayer::GetInputDepth() const
{
    return InputSize() / (_windowSizeX * _windowSizeY);
}

void WriteStruct(const CStructWriter &writer, const ConvoLayer &layer)
{
	WriteStruct(writer, (const SingleInputLayer &)layer);

	writer("windowSizeX", layer._windowSizeX)
		  ("windowSizeY", layer._windowSizeY)
		  ("strideX", layer._strideX)
		  ("strideY", layer._strideY)
		  ("padWidth", layer._padWidth)
		  ("padHeight", layer._padHeight)
		  ("gradConsumer", layer._gradConsumer)
		  ("inputDepth", layer.GetInputDepth())
		  ("outputDepth", layer.OutputSize())
		  ;
}

void ReadStruct(const CStructReader &reader, ConvoLayer &layer)
{
	size_t outputDepth, inputDepth;

	reader("windowSizeX", layer._windowSizeX)
		  ("windowSizeY", layer._windowSizeY)
		  ("strideX", layer._strideX)
		  ("strideY", layer._strideY)
		  ("padWidth", layer._padWidth)
		  ("padHeight", layer._padHeight)
		  ("inputDepth", inputDepth)
		  ("outputDepth", outputDepth)
		  ("gradConsumer", layer._gradConsumer)
		  ;

	layer._weights = CWeights(layer._windowSizeX * layer._windowSizeY * inputDepth,
							  outputDepth);

	// Don't use WeightLayer's read function because this
	// read already grabs all of the necessary information
	// for it
	ReadStruct(reader, (SingleInputLayer&)layer);
}

AXON_SERIALIZE_DERIVED_TYPE(ILayer, ConvoLayer, ConvoLayer);


