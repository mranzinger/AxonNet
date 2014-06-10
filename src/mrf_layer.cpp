/*
 * mrf_layer.cpp
 *
 *  Created on: May 23, 2014
 *      Author: mike
 */

#include "mrf_layer.h"

#include "neural_net.h"
#include "thread/parallel_for.h"

using namespace std;

MRFLayer::MRFLayer(std::string name,
				   uint32_t width, uint32_t height)
	: MRFLayer(move(name), "", width, height)
{
}

MRFLayer::MRFLayer(std::string name, std::string inputName,
		uint32_t width, uint32_t height)
	: LayerBase(move(name)), _inputName(move(inputName)),
	  _width(width), _height(height)
{
}

void MRFLayer::Compute(ParamMap& inputMap, bool isTraining)
{
	const Params &input = *GetData(inputMap, GetInputName());

	// TODO: Lift this restriction
	assert(input.Width >= _width &&
		   input.Height >= _height);

	Params output(_width, _height, input.Depth,
			      new CMatrix(_width * _height * input.Depth, input.Cols));

	Params coordsOut(1, 2, input.Depth,
					 new CMatrix(2 * input.Depth, input.Cols));

	const CMatrix &mInput = input.GetHostMatrix();

	CMatrix &mOutput = output.GetHostMatrix();
	CMatrix &mCoords = coordsOut.GetHostMatrix();

	FastFor(GetThreadPool(), 0u, input.Cols, 1u,
		[&, this] (int imageIdx)
	{
		RMap outputField(mOutput.data() + imageIdx * mOutput.size(),
						 _height, _width * input.Depth);

		const RMap inputField(const_cast<Real*>(mInput.data()) + imageIdx * mInput.size(),
						input.Height, input.Width * input.Depth);

		UMapVector vecCoords(mCoords.data() + imageIdx * 2 * input.Depth,
							 2 * input.Depth);

		// A summed area table is the most efficient way to compute the
		// windowed sum over a scanned region. The only weird thing about
		// this over a traditional sat is that the depths are summed independently
		RMatrix sumAreaTable = RMatrix::Zero(input.Height, input.Width * input.Depth);

		CalcSAT(inputField, sumAreaTable, input.Depth);

		vector<Real> maxes(input.Depth, numeric_limits<Real>::lowest());
		vector<pair<int, int>> coords(input.Depth);

		for (int row = 0, rowEnd = input.Height - _height + 1; row < rowEnd; ++row)
		{
			int rowBottom = row + _height - 1;

			for (int col = 0, colEnd = input.Width - _width + 1; col < colEnd; ++col)
			{
				int colStart = col * input.Depth;
				int colRight = colStart + (_width - 1) * input.Depth;

				for (int layer = 0; layer < input.Depth; ++layer)
				{
					const Real a = row > 0 && col > 0 ?
										sumAreaTable(row - 1, colStart + layer - input.Depth)
									:   0.0f;
					const Real b = row > 0 ?
							            sumAreaTable(row - 1, colRight + layer)
							        :   0.0f;
					const Real c = col > 0 ?
										sumAreaTable(rowBottom, colStart + layer - input.Depth)
									:   0.0f;
					const Real d = sumAreaTable(rowBottom, colRight + layer);

					const Real sum = a + d - b - c;

					Real &currMax = maxes[layer];
					if (sum > currMax)
					{
						currMax = sum;
						coords[layer] = make_pair(row, col);
					}
				}
			}
		}

		// Pulling all of the disparate fields together is a slightly
		// annoying operation
		for (int row = 0; row < _height; ++row)
		{
			for (int col = 0; col < _width; ++col)
			{
				for (int layer = 0; layer < input.Depth; ++layer)
				{
					const pair<int, int> &basePt = coords[layer];

					const int srcRow = basePt.first + row;
					const int srcCol = (basePt.second + col) * input.Depth + layer;

					const Real val = inputField(srcRow, srcCol);

					const int destCol = col * input.Depth + layer;

					outputField(row, destCol) = val;
				}
			}
		}

		// Write out the coordinates
		for (int i = 0; i < coords.size(); ++i)
		{
			const pair<int,int> &coord = coords[i];
			vecCoords(i * 2) = coord.first;
			vecCoords(i * 2 + 1) = coord.second;
		}
	});

	inputMap[_name] = move(output);
	inputMap[_name + "-coords"] = move(coordsOut);
}

void MRFLayer::Backprop(const ParamMap& computeMap, ParamMap& inputErrorMap)
{
	const string &inputName = GetInputName();

	const Params &lastInput = *GetData(computeMap, inputName);
	const Params &outputErrs = *GetData(inputErrorMap, _name);
	const Params &outputCoords = *GetData(computeMap, _name + "-coords");

	Params inputErrors(lastInput,
			  new CMatrix(CMatrix::Zero(lastInput.Rows, lastInput.Cols)));

	const CMatrix &mLastInput = lastInput.GetHostMatrix();
	const CMatrix &mOutputErrs = outputErrs.GetHostMatrix();
	const CMatrix &mOutputCoords = outputCoords.GetHostMatrix();

	CMatrix &mInputErrors = inputErrors.GetHostMatrix();

	FastFor(GetThreadPool(), 0u, lastInput.Cols, 1u,
		[&, this] (int imageIdx)
	{
		const RMap imgOutputErrs(const_cast<Real*>(mOutputErrs.data()) + imageIdx * mOutputErrs.size(),
						   outputErrs.Height, outputErrs.Width * outputErrs.Depth);
		const UMapVector imgOutputCoords(const_cast<Real*>(mOutputCoords.data()) + imageIdx * mOutputCoords.size(),
						   mOutputCoords.size());

		RMap imgInputErrs(mInputErrors.data() + imageIdx * mInputErrors.size(),
						  inputErrors.Height, inputErrors.Width * inputErrors.Depth);

		vector<pair<int,int>> iCoords(lastInput.Depth);
		for (int i = 0; i < imgOutputCoords.size(); i += 2)
			iCoords.emplace_back(imgOutputCoords(i), imgOutputCoords(i + 1));

		for (int row = 0; row < outputErrs.Height; ++row)
		{
			for (int col = 0; col < outputErrs.Width; ++col)
			{
				for (int layer = 0; layer < lastInput.Depth; ++layer)
				{
					const pair<int, int> &baseCoord = iCoords[layer];

					const int opCol = col * lastInput.Depth + layer;

					const Real opVal = imgOutputErrs(row, opCol);

					const int ipRow = baseCoord.first + row;
					const int ipCol = (baseCoord.second + col) * lastInput.Depth + layer;

					imgInputErrs(ipRow, ipCol) = opVal;
				}
			}
		}
	});

	inputErrorMap[inputName] = move(inputErrors);
}

template<typename Function>
void CalcSATFn(const RMap &inputField, RMatrix &sumAreaTable, uint32_t depth, Function fn)
{
	// The good news is that computing this is a separable problem. I just
	// don't feel like doing that right now
	for (int row = 0; row < inputField.rows(); ++row)
	{
		for (int col = 0; col < inputField.cols(); col += depth)
		{
			for (int dCell = col; dCell < col + depth; ++dCell)
			{
				const Real above = row > 0 ? sumAreaTable(row - 1, dCell) : 0.0f;
				const Real left = dCell >= depth ? sumAreaTable(row, dCell - depth) : 0.0f;
				const Real al = row > 0 && dCell >= depth ? sumAreaTable(row - 1, dCell - depth) : 0.0f;
				const Real me = fn( inputField(row, dCell) );

				const Real val = above + left + me - al;

				sumAreaTable(row, dCell) = val;
			}
		}
	}
}

void MRFLayer::CalcSAT(const RMap& inputField, RMatrix& sumAreaTable, uint32_t depth) const
{
	switch (_function)
	{
	case MRFFunction::Default:
	default:
		CalcSATFn(inputField, sumAreaTable, depth, [] (Real val) { return val; });
		break;
	case MRFFunction::Abs:
		CalcSATFn(inputField, sumAreaTable, depth, [] (Real val) { return abs(val); });
		break;
	case MRFFunction::Squared:
		CalcSATFn(inputField, sumAreaTable, depth, [] (Real val) { return Square(val); });
		break;
	case MRFFunction::SignSquared:
		CalcSATFn(inputField, sumAreaTable, depth,
			[] (Real val)
			{
				Real sqVal = Square(val);

				if (val < 0)
					sqVal = -sqVal;

				return sqVal;
			});
		break;
	}
}

const std::string& MRFLayer::GetInputName()
{
	if (!_inputName.empty())
		return _inputName;

	int myIdx = _net->GetLayerIndex(this);

	assert(myIdx >= 0);

	if (myIdx == 0)
		_inputName = ITrainProvider::DEFAULT_INPUT_NAME;
	else
		_inputName = _net->GetLayer(myIdx - 1)->GetLayerName();

	return _inputName;
}

ENUM_IO_MAP(MRFFunction)
	DEFMAP(MRFFunction::Default, "default")
	ENMAP(MRFFunction::Abs, "abs")
	ENMAP(MRFFunction::Squared, "squared")
	ENMAP(MRFFunction::SignSquared, "sign-squared");

void BindStruct(const aser::CStructBinder &binder, MRFLayer &layer)
{
	binder("width", layer._width)
		  ("height", layer._height)
		  ("function", layer._function)
		  ("inputName", layer._inputName);

	BindStruct(binder, (LayerBase&)layer);
}

AXON_SERIALIZE_DERIVED_TYPE(ILayer, MRFLayer, MRFLayer);


