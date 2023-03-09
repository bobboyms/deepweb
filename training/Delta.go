package training

func Mult(nextDelta, activation float64) float64 {
	return activation * nextDelta
}

func CalculateOutputDelta(desiredOutput []float64, realOutput []float64) []float64 {
	delta := make([]float64, len(realOutput))

	for i := 0; i < len(realOutput); i++ {
		delta[i] = CalculateOutputError(desiredOutput[i], realOutput[i]) * SigmoidDerivative(realOutput[i])
	}

	return delta
}

func CalculateHiddenDelta(nextDelta, weights, activation []float64) []float64 {
	hiddenDelta := make([]float64, len(weights))
	for i := 0; i < len(weights); i++ {
		deltaSum := 0.0
		for j := 0; j < len(nextDelta); j++ {
			deltaSum += nextDelta[j] * weights[i]
		}
		hiddenDelta[i] = deltaSum * SigmoidDerivative(activation[i])
	}
	return hiddenDelta
}
