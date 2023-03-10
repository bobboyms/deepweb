package test

import (
	"deepgo/mtx"
	"deepgo/stepfunction"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestSoftMax(t *testing.T) {
	//result := stepfunction.SoftmaxFunction([]float64{1, 2, 3, 23, 6})
	//sum := 0.0
	//for _, number := range result {
	//	sum += number
	//}
	//
	//if sum != 1 {
	//	t.Fatalf("Expected 1 found %f", sum)
	//}
}

func TestSigmoid(t *testing.T) {
	result := stepfunction.NewSigmoid().Activation(mat.NewDense(1, 1, []float64{0}))
	slice := mtx.DenseToSlice(result)[0][0]

	if slice != 0.5 {
		t.Fatalf("Expected 0.5 found %v", slice)
	}
}
