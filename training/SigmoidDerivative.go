package training

import "math"

func SigmoidDerivative(x float64) float64 {
	return math.Exp(-x) / math.Pow(1+math.Exp(-x), 2)
}
