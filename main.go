package main

import (
	"deepgo/layers"
	"deepgo/models"
	"deepgo/stepfunction"
	"gonum.org/v1/gonum/mat"
)

func main() {

	input := mat.NewDense(3, 2, []float64{
		0.0001, 0.0002, 0.00003,
		0.0004, 0.0005, 0.00006})

	model := models.NewSequential()
	row, col := input.Dims()

	model.AddLayer(layers.NewDense(
		layers.Dims{Rows: row, Cols: col},
		layers.Dims{Rows: 5, Cols: 5},
		stepfunction.NewRelu()))

	model.AddLayer(layers.NewDense(
		layers.Dims{Rows: 5, Cols: 5},
		layers.Dims{Rows: 25, Cols: 30},
		stepfunction.NewRelu()))

	model.AddLayer(layers.NewDense(
		layers.Dims{Rows: 25, Cols: 30},
		layers.Dims{Rows: 25, Cols: 30},
		stepfunction.NewRelu()))

	model.AddLayer(layers.NewDense(
		layers.Dims{Rows: 25, Cols: 30},
		layers.Dims{Rows: 150, Cols: 100},
		stepfunction.NewRelu()))

	model.AddLayer(layers.NewDense(
		layers.Dims{Rows: 150, Cols: 100},
		layers.Dims{Rows: 2, Cols: 1},
		stepfunction.NewRelu()))

	model.Compile(input)
	//fmt.Printf("\n%v\n", mat.Formatted(output))

}

func dotProduct(a, b *mat.Dense) {

	output := &mat.Dense{}
	output.Mul(a, b)
}
