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
	result := stepfunction.NewSigmoid().Activation(0)
	fmt.Println(result)

	if result != 0.5 {
		t.Fatalf("Expected 0.5 found %f", result)
	}
}
