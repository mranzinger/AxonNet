/*
 * mrf_layer.cpp
 *
 *  Created on: May 23, 2014
 *      Author: mike
 */

#include "mrf_layer.h"

using namespace std;

MRFLayer::MRFLayer(std::string name,
				   size_t width, size_t height)
	: SingleInputLayer(move(name)),
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
			      CMatrix(_width * _height * input.Depth, input.BatchSize()));

	Params coordsOut(1, 2, input.Depth,
					 CMatrix(2 * input.Depth, input.BatchSize()));

	for (int imageIdx = 0; imageIdx < input.BatchSize(); ++imageIdx)
	{
		CMap outputField(output.Data.data() + imageIdx * output.size(),
						 _height, _width * input.Depth);

		RMap inputField(const_cast<Real*>(input.Data.data()) + imageIdx * input.size(),
						input.Height, input.Width * input.Depth);

		// A summed area table is the most efficient way to compute the
		// windowed sum over a scanned region. The only weird thing about
		// this over a traditional sat is that the depths are summed independently
		RMatrix sumAreaTable = RMatrix::Zero(input.Height, input.Width * input.Depth);

		CalcSAT(inputField, sumAreaTable, input.Depth);

		vector<Real> maxes(input.Depth, numeric_limits<Real>::lowest());
		vector<pair<int, int>> coords(input.Depth);

		for (int row = 0, rowEnd = input.Height - _height; row < rowEnd; ++row)
		{
			int rowBottom = row + _height - 1;

			for (int col = 0, colEnd = input.Width - _width; col < colEnd; ++col)
			{
				int colStart = col * input.Depth;
				int colRight = colStart + (_width - 1) * input.Depth;

				for (int layer = 0; layer < input.Depth; ++layer)
				{
					const Real a = sumAreaTable(row, colStart + layer);
					const Real b = sumAreaTable(row, colRight + layer);
					const Real c = sumAreaTable(rowBottom, colStart + layer);
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

				}
			}
		}
	}

	inputMap[_name] = move(output);
	inputMap[_name + "-coords"] = move(coordsOut);
}

void MRFLayer::CalcSAT(const RMap& inputField, RMatrix& sumAreaTable, int depth) const
{
	// The good news is that computing this is a separable problem. I just
	// don't feel like doing that right now
	for (int row = 0; row < inputField.rows(); ++row)
	{
		for (int col = 0; col < inputField.cols(); col += depth)
		{
			for (int dCell = col; dCell < col + depth; ++dCell)
			{
				const Real above = row > 0 ? inputField(row - 1, dCell) : 0.0f;
				const Real left = dCell > depth ? inputField(row, dCell - depth) : 0.0f;
				const Real me = inputField(row, dCell);

				const Real val = above + left + me;

				sumAreaTable(row, dCell) = val;
			}
		}
	}
}

void MRFLayer::Backprop(const ParamMap& computeMap, ParamMap& inputErrorMap)
{
	throw runtime_error("Not implemented yet.");
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

void BindStruct(const aser::CStructBinder &binder, MRFLayer &layer)
{
	BindStruct(binder, (SingleInputLayer&)layer);

	binder("width", layer._width)
		  ("height", layer._height);
}

AXON_SERIALIZE_DERIVED_TYPE(ILayer, MRFLayer, MRFLayer);


