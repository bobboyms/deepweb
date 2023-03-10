package models

import (
	"deepgo/layers"
	"deepgo/mtx"
	"deepgo/training"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math/rand"
	"time"
)

type Sequential struct {
	Count   int
	Weights []*mat.Dense
	Layers  []layers.Layer
	//Outputs []*mat.Dense
}

func NewSequential(weights []*mat.Dense) *Sequential {
	return &Sequential{
		Count:   0,
		Weights: weights,
		Layers:  make([]layers.Layer, 0),
	}
}

func (s *Sequential) AddLayer(layer layers.Layer) {

	if s.Weights == nil {
		dense := layer.(*layers.Dense)
		layer.AdjustWeights(CreateWeights(dense.InSize, dense.OutSize))
		s.Layers = append(s.Layers, layer)
	} else {
		layer.AdjustWeights(s.Weights[s.Count])
		s.Layers = append(s.Layers, layer)
		s.Count += 1
	}

}

func (s *Sequential) FeedForward(input []float64) ([][]float64, [][]float64) {

	outputsLayer := make([][]float64, len(s.Layers))
	inputsLayer := make([][]float64, len(s.Layers))

	layer := s.Layers[0]

	inputsLayer[0] = input
	output := layer.Process(input)
	outputsLayer[0] = output.RawMatrix().Data
	for i, layer := range s.Layers[1:] {
		inputsLayer[i+1] = output.RawMatrix().Data
		output = layer.Process(output.RawMatrix().Data)
		outputsLayer[i+1] = output.RawMatrix().Data
	}

	return outputsLayer, inputsLayer
}

func DeltaCalculation(outputs [][]float64, realOutput *mat.Dense, weights []*mat.Dense) [][]float64 {
	first := true
	length := len(outputs)
	var deltaLayers [][]float64
	var outputDeltaLayer []float64
	for i := length - 1; i >= 0; i-- {
		outputLayer := outputs[i]
		if first {
			outputDeltaLayer = training.CalculateOutputDelta(outputLayer, realOutput.RawMatrix().Data)
			deltaLayers = append(deltaLayers, outputDeltaLayer)

			//weights := mtx.DenseToSlice(weights[i])
			//outputDeltaLayer = training.CalculateHiddenDelta(outputDeltaLayer, weights, outputLayer)
			//deltaLayers = append(deltaLayers, outputDeltaLayer)

			first = false
		} else {
			weights := mtx.DenseToSlice(mtx.TransposeToDense(weights[i].T()))
			outputDeltaLayer = training.CalculateHiddenDelta(outputDeltaLayer, weights, outputLayer)
			deltaLayers = append(deltaLayers, outputDeltaLayer)
		}

	}
	return deltaLayers
}

func StartTraining(sequential *Sequential, x, y *mat.Dense, epochs int, lr float64) {

	layersSize := len(sequential.Layers)

	for i := 0; i < epochs; i++ {
		xRow, _ := x.Dims()
		_, yCol := y.Dims()

		var deltasClass [][][]float64
		var inputsClass [][][]float64
		errors := make([]float64, xRow)
		for r := 0; r < xRow; r++ {

			dataTraining := x.RowView(r).(*mat.VecDense).RawVector().Data
			dataPredict := y.RowView(r).(*mat.VecDense).RawVector().Data

			outputsLayer, inputsLayer := sequential.FeedForward(dataTraining)

			errors[r] = training.MeanAbsoluteError(dataPredict, outputsLayer[layersSize-1])
			fmt.Println("Errors: ", errors[r])
			deltaLayers := DeltaCalculation(outputsLayer, mat.NewDense(1, yCol, dataPredict), sequential.Weights)

			deltasClass = append(deltasClass, deltaLayers)
			inputsClass = append(inputsClass, ReverseArray(inputsLayer))
		}

		//fmt.Printf("Mean error: %.4f\n", Mean(errors))

		sizeClass := len(y.RawMatrix().Data)
		layersMultiClass := make([][][]float64, sizeClass)
		for class := 0; class < sizeClass; class++ {
			deltas := deltasClass[class]
			inputs := inputsClass[class]

			layerSize := len(deltas)
			mults := make([][]float64, layerSize)
			for layer := 0; layer < layerSize; layer++ {
				result := training.ActivationDotDelta(inputs[layer], deltas[layer])
				mults[layer] = result
			}

			layersMultiClass[class] = mults

		}

		multiply := func(classSum []float64, weights [][]float64) *mat.Dense {

			rows, cols := len(weights), len(weights[0])
			flat := make([]float64, rows*cols)

			for i, row := range weights {
				for j, col := range row {
					flat[i*cols+j] = (col * 1) + classSum[i]*lr
				}
			}

			return mat.NewDense(rows, cols, flat)

		}

		layersSumClass := mtx.Add(layersMultiClass)
		newWeights := make([]*mat.Dense, len(layersSumClass))
		for i, weighs := range ReverseDense(sequential.Weights) {
			newWeights[i] = multiply(layersSumClass[i], mtx.DenseToSlice(weighs))
		}

		newSequential := NewSequential(ReverseDense(newWeights))
		newSequential.Layers = sequential.Layers
		sequential = newSequential

		//fmt.Println(">>>> Epoca:", i)
		//fmt.Println("Camadas:", len(sequential.Layers))
		//fmt.Println("Pesos:", mtx.DenseToSlice(sequential.Weights[1]))

	}

}

func CreateWeights(row, col int) *mat.Dense {

	numberWeights := row * col

	values := make([]float64, numberWeights)
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < numberWeights; i++ {
		values[i] = rand.Float64() / 1000

	}
	return mat.NewDense(row, col, values)
}

func Mean(nums []float64) float64 {
	sum := 0.0
	for _, num := range nums {
		sum += num
	}
	return sum / float64(len(nums))
}

func ReverseArray(data [][]float64) [][]float64 {
	length := len(data)
	var result [][]float64
	for i := length - 1; i >= 0; i-- {
		result = append(result, data[i])
	}
	return result
}

func ReverseDense(data []*mat.Dense) []*mat.Dense {
	length := len(data)
	var result []*mat.Dense
	for i := length - 1; i >= 0; i-- {
		result = append(result, data[i])
	}
	return result
}

func GetRow(mtx *mat.Dense, line int) []float64 {
	_, col := mtx.Dims()
	row := mtx.RowView(line)

	data := make([]float64, col)
	for x := 0; x < col; x++ {
		data[x] = row.AtVec(x)
	}
	return data
}
