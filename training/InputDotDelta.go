package training

func InputDotDelta(inputs, deltas []float64) []float64 {
	results := make([]float64, len(inputs))

	for i, input := range inputs {
		sum := 0.0
		for _, delta := range deltas {
			sum += input * delta
		}
		results[i] = sum
	}

	return results
}
