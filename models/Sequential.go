package models

import (
	"deepgo/layers"
	"deepgo/training"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math/rand"
	"time"
)

type Sequential struct {
	Layers  []layers.Layer
	Outputs []*mat.Dense
}

func NewSequential() *Sequential {
	return &Sequential{
		Layers: make([]layers.Layer, 0),
	}
}

func (s *Sequential) AddLayer(layer layers.Layer) {
	dense := layer.(*layers.Dense)
	layer.AdjustWeights(CreateWeights(dense.Inputs, dense.Outputs))
	s.Layers = append(s.Layers, layer)
}

func (s *Sequential) FeedForward(input []float64) *mat.Dense {

	s.Outputs = make([]*mat.Dense, len(s.Layers))

	layer := s.Layers[0]
	output := layer.Process(input)
	s.Outputs[0] = output
	for i, layer := range s.Layers[1:] {
		output = layer.Process(output.RawMatrix().Data)
		s.Outputs[i+1] = output
	}

	return output
}

func (s *Sequential) Backpropagation(realOutput *mat.Dense) {
	length := len(s.Outputs)
	var outputDeltaLayer []float64
	first := true
	for i := length - 1; i > 0; i-- {
		outputLayer := s.Outputs[i]
		if first {
			outputDeltaLayer = training.CalculateOutputDelta(outputLayer.RawMatrix().Data, realOutput.RawMatrix().Data)

			s.Layers[i].GetWeights()
			outputDeltaLayer = training.CalculateHiddenDelta(outputDeltaLayer, nil, outputLayer.RawMatrix().Data)

			//fmt.Println("Out 2: ", outputDeltaLayer)
			//fmt.Println("Out 1: ", outputDeltaLayer)
			//
			first = false
		}

		outputDeltaLayer = training.CalculateHiddenDelta(outputDeltaLayer, nil, outputLayer.RawMatrix().Data)
		//fmt.Println("Out 2: ", outputDeltaLayer)

	}
}

func StartTraining(sequential *Sequential, x, y *mat.Dense, epochs int, lr float64) {

	for i := 0; i < epochs; i++ {
		xRow, _ := x.Dims()
		_, yCol := y.Dims()

		errors := make([]float64, xRow)
		for r := 0; r < xRow; r++ {
			dataTraining := x.RowView(r).(*mat.VecDense).RawVector().Data
			dataPredict := y.RowView(r).(*mat.VecDense).RawVector().Data

			output := sequential.FeedForward(dataTraining)
			errors[r] = training.MeanAbsoluteError(dataPredict, output.RawMatrix().Data)
			sequential.Backpropagation(mat.NewDense(1, yCol, dataPredict))
		}

		fmt.Printf("Mean error: %.4f\n", Mean(errors))

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

func GetRow(mtx *mat.Dense, line int) []float64 {
	_, col := mtx.Dims()
	row := mtx.RowView(line)

	data := make([]float64, col)
	for x := 0; x < col; x++ {
		data[x] = row.AtVec(x)
	}
	return data
}
