package test

import (
	"deepgo/layers"
	"deepgo/models"
	"deepgo/stepfunction"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestLayers(t *testing.T) {

	input := mat.NewDense(4, 3, []float64{
		1, 0.025, 3,
		-4, 5, -6,
		68, 500, -6,
		24, -8745, -6})

	model := models.NewSequential()
	row, col := input.Dims()

	model.AddLayer(layers.NewDense(
		layers.Dims{Rows: row, Cols: col},
		layers.Dims{Rows: 50, Cols: 25},
		stepfunction.NewSigmoid()))

	model.AddLayer(layers.NewDense(
		layers.Dims{Rows: 50, Cols: 25},
		layers.Dims{Rows: 3, Cols: 3},
		stepfunction.NewRelu()))

	model.AddLayer(layers.NewDense(
		layers.Dims{Rows: 3, Cols: 3},
		layers.Dims{Rows: 1, Cols: 1},
		stepfunction.NewSoftmax()))

	layer := model.Layers[0]
	output := layer.Process(input)
	for _, layer := range model.Layers[1:] {
		output = layer.Process(output)
	}

	re := output.RawMatrix().Data[0]

	if re != 1 {
		t.Fatalf("Expected 1 found %f", re)
	}

}
