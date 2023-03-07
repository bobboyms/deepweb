package main

import (
	"deepgo/layers"
	"deepgo/models"
	"deepgo/stepfunction"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
)

func softmax(x []float64) []float64 {
	var sum float64
	for _, v := range x {
		sum += math.Exp(v)
	}
	result := make([]float64, len(x))
	for i, v := range x {
		result[i] = math.Exp(v) / sum
	}
	return result
}

func main() {

	fmt.Println("P: ", softmax([]float64{0, 0.2, 0.1}))

	input := mat.NewDense(3, 2, []float64{
		1, 0, 3, -4, 5, -6})

	model := models.NewSequential()
	row, col := input.Dims()

	model.AddLayer(layers.NewDense(
		layers.Dims{Rows: row, Cols: col},
		layers.Dims{Rows: 50, Cols: 25},
		stepfunction.NewRelu()))

	model.AddLayer(layers.NewDense(
		layers.Dims{Rows: 50, Cols: 25},
		layers.Dims{Rows: 3, Cols: 3},
		stepfunction.NewRelu()))

	model.AddLayer(layers.NewDense(
		layers.Dims{Rows: 3, Cols: 3},
		layers.Dims{Rows: 1, Cols: 1},
		stepfunction.NewSoftmax()))

	model.Compile(input)

}

func dotProduct(a, b *mat.Dense) {

	output := &mat.Dense{}
	output.Mul(a, b)
}
