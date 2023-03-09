package main

import (
	"deepgo/layers"
	"deepgo/models"
	"deepgo/stepfunction"
	"gonum.org/v1/gonum/mat"
)

func main() {

	inputs := mat.NewDense(4, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	})

	outputs := mat.NewDense(4, 1, []float64{
		0,
		0,
		0,
		1,
	})

	//weights := mat.NewDense(2, 3, []float64{
	//	-0.424, -0.740, -0.961,
	//	0.358, -0.577, -0.469,
	//})

	activation := stepfunction.NewSigmoid()

	model := models.NewSequential()
	model.AddLayer(layers.NewDense(2, 3, activation))
	model.AddLayer(layers.NewDense(3, 1, activation))

	//recordInput := inputs.RowView(2).(*mat.VecDense).RawVector().Data
	//output := model.FeedForward(recordInput)
	//output := dense.Process(recordInput)

	models.StartTraining(model, inputs, outputs, 1, 0.01)

	//fmt.Printf("Output \n%v\n", mat.Formatted(output))
}
