package training

import (
	"deepgo/mtx"
)

func ActivationDotDelta(activation []float64, nextDelta []float64) []float64 {

	size := len(activation)
	dotValues := make([]float64, size)
	for i, value := range activation {
		sum := 0.0
		for _, delta := range nextDelta {
			sum += value * delta
		}
		dotValues[i] = sum
	}

	return dotValues
}

func CalculateOutputDelta(desiredOutput []float64, realOutput []float64) []float64 {
	size := len(realOutput)
	delta := make([]float64, size)

	for i := 0; i < size; i++ {
		delta[i] = CalculateOutputError(desiredOutput[i], realOutput[i]) * SigmoidDerivative(realOutput[i])
	}
	return delta
}

func CalculateHiddenDelta(outputDeltas []float64, weights [][]float64, activation []float64) []float64 {
	size := len(activation)
	deltas := make([]float64, size)
	sumWeights := mtx.SumCol(weights)
	for j := 0; j < size; j++ {
		var sum float64
		for i := 0; i < len(outputDeltas); i++ {
			sum += SigmoidDerivative(activation[j]) * sumWeights[j] * outputDeltas[i]
		}
		deltas[j] = sum
	}
	return deltas
}
