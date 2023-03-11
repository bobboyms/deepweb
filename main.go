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

	weights := []*mat.Dense{
		mat.NewDense(2, 3, []float64{
			-0.424, -0.740, -0.961,
			0.358, -0.577, -0.469,
		}),
		mat.NewDense(3, 1, []float64{
			-0.017,
			-0.893,
			0.148,
		}),
	}

	model := models.NewSequential(weights)
	model.AddLayer(layers.NewDense(2, 3, stepfunction.NewSigmoid()))
	model.AddLayer(layers.NewDense(3, 1, stepfunction.NewSigmoid()))

	models.StartTraining(model, inputs, outputs, 1000000, 0.5)

}
