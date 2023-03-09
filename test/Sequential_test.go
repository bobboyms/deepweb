package test

import (
	"testing"
)

func TestSequential(t *testing.T) {

	//input := mat.NewDense(3, 2, []float64{
	//	-10, 0.0002, 9.00003,
	//	0.0004, 3.0005, 0.25})
	//
	//model := models.NewTrainSequential(
	//	training.Adam
	//	training.MeanAbsoluteError)
	//row, col := input.Dims()
	//
	//model.AddLayer(layers.NewDense(
	//	layers.Dims{Rows: row, Cols: col},
	//	layers.Dims{Rows: 500, Cols: 50},
	//	stepfunction.NewSigmoid()))
	//
	//model.AddLayer(layers.NewDense(
	//	layers.Dims{Rows: 500, Cols: 50},
	//	layers.Dims{Rows: 3, Cols: 3},
	//	stepfunction.NewRelu()))
	//
	//model.AddLayer(layers.NewDense(
	//	layers.Dims{Rows: 3, Cols: 3},
	//	layers.Dims{Rows: 1, Cols: 1},
	//	stepfunction.NewSoftmax()))
	//
	//model.FeedForward(input)
	//
	//layer := model.Layers[0]
	//output := layer.Process(input)
	//for _, layer := range model.Layers[1:] {
	//	output = layer.Process(output)
	//}
	//
	////fmt.Printf("\n%v\n", mat.Formatted(output))
	//re := output.RawMatrix().Data[0]
	//
	//if re != 1 {
	//	t.Fatalf("Expected 1 found %f", re)
	//}

}
