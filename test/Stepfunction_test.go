package test

import (
	"deepgo/stepfunction"
	"fmt"
	"testing"
)

func TestSoftMax(t *testing.T) {
	result := stepfunction.SoftmaxFunction([]float64{1, 2, 3, 23, 6})
	sum := 0.0
	for _, number := range result {
		sum += number
	}

	if sum != 1 {
		t.Fatalf("Expected 1 found %f", sum)
	}
}

func TestSigmoid(t *testing.T) {
	result := stepfunction.NewSigmoid().Activation(1)
	fmt.Println(result)

	if result != 0.7310585786300049 {
		t.Fatalf("Expected 1 found %f", result)
	}
}
