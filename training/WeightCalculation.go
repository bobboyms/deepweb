package training

func WeightAdjustmentCalculation(nextDelta [][]float64, activations []float64, oldWeight, momentum, learningRate float64) float64 {
	result := 0.0
	for i, activation := range activations {

		deltaList := nextDelta[i]

		sum := 0.0
		for _, delta := range deltaList {
			sum += activation * delta
		}
		result += sum
	}

	return (oldWeight * momentum) + result*learningRate
}
