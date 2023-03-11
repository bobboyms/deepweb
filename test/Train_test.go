package test

import (
	"deepgo/mtx"
	"deepgo/training"
	"fmt"
	"math"
	"testing"
)

func EQ(array1, array2 []float64) bool {
	equal := true
	for i := 0; i < len(array1); i++ {
		if array1[i] != array2[i] {
			equal = false
			break
		}
	}

	if equal {
		return true
	} else {
		return false
	}
}

func RoundTo3DecimalPlaces(num float64) float64 {
	factor := math.Pow(10, 3)
	rounded := math.Round(num*factor) / factor
	return rounded
}

func TestWeightAdjustmentCalculation(t *testing.T) {
	//learningRate := 0.3
	//momentum := 1.0
	//nextDelta := [][]float64{{-0.098}, {0.139}, {0.139}, {-0.114}}
	//activations := []float64{0.5, 0.360, 0.323, 0.211}
	//weight := -0.893
	//
	//result := training.WeightAdjustmentCalculation(nextDelta, activations, weight, momentum, learningRate)
	//
	//if result != -0.8864351 {
	//	t.Fatalf("Expected -0.8864351 found %f", result)
	//}
	//
	//fmt.Println(result)

}

func TestCalculateHiddenDelta(t *testing.T) {
	nextDelta := []float64{-0.098}
	activations := []float64{0.5, 0.5, 0.5}
	weights := [][]float64{{0.017}, {-0.893}, {0.148}}

	hiddenDelta := training.CalculateHiddenDelta(nextDelta, weights, activations)
	fmt.Println(hiddenDelta)

	results := []float64{-0.00039151618452785644, 0.020566114869610342, -0.0034084938417719268}
	if !EQ(results, hiddenDelta) {
		t.Fatalf("Expected -0.00039151618452785644, 0.020566114869610342, -0.0034084938417719268 found %f", hiddenDelta)
	}

}

func TestCalculateOutputDelta(t *testing.T) {
	desiredOutput := []float64{0}
	realOutput := []float64{0.406}
	result := training.CalculateOutputDelta(desiredOutput, realOutput)

	if result[0] != -0.09742956989447803 {
		t.Fatalf("Expected -0.09742956989447803 found %v", result)
	}
}

func TestCalculateOutputError(t *testing.T) {
	result := training.CalculateOutputError(1, 1)

	if result != 0 {
		t.Fatalf("Expected 0 found %f", result)
	}

	result = training.CalculateOutputError(2, 1)
	if result != 1 {
		t.Fatalf("Expected 0 found %f", result)
	}
}

func TestSigmoidDerivative(t *testing.T) {

	result := training.SigmoidDerivative(0.5)
	fmt.Println(result)

	if result != 0.2350037122015945 {
		t.Fatalf("Expected 0.2350037122015945 found %f", result)
	}

}

func TestMeanAbsoluteError(t *testing.T) {
	ytrue := []float64{0, 0, 0, 1}
	predict := []float64{0, 0, 0, 1}

	result := training.MeanAbsoluteError(ytrue, predict)

	if result != 0 {
		t.Fatalf("Expected 0 found %f", result)
	}

	ytrue = []float64{0, 0, 0, 1}
	predict = []float64{0, 1, 0, 1}

	result = training.MeanAbsoluteError(ytrue, predict)

	if result != 0.25 {
		t.Fatalf("Expected 0.25 found %f", result)
	}

	ytrue = []float64{0, 0, 0, 1}
	predict = []float64{1, 1, 1, 0}

	result = training.MeanAbsoluteError(ytrue, predict)

	if result != 1 {
		t.Fatalf("Expected 1 found %f", result)
	}

}

func TestActivationDotDelta(t *testing.T) {

	delta := []float64{2}
	activations := []float64{0.5, 0.5, 0.5}
	result := training.ActivationDotDelta(activations, delta)

	if !EQ(result, []float64{1, 1, 1}) {
		t.Fatalf("Expected [1 1 1] found %v", result)
	}
}

func TestAdd(t *testing.T) {
	datas := [][][]float64{{{1, 1, 1}, {0, 0}}, {{1, 1, 1}, {1, 1}}}
	mtx.Add(datas)
}
