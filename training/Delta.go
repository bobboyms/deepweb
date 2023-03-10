package training

import (
	"deepgo/mtx"
)

func Mult(nextDelta, activation float64) float64 {
	return activation * nextDelta
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
