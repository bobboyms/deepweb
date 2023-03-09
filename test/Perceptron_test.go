package test

import (
	"deepgo/neurons"
	"deepgo/stepfunction"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestProcess(t *testing.T) {
	input := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	weights := mat.NewDense(2, 2, []float64{2, 2, 2, 2})

	stepFunction := stepfunction.NewSoftmax()
	perceptron := neurons.NewPerceptron(weights, stepFunction, 0.1)
	output := perceptron.Process(input)

	if output != 20.1 {
		t.Fatalf("Expected 20.1 found %f", output)
	}

}
