package training

import (
	"math"
)

func MeanAbsoluteError(trueValues, predictValues []float64) float64 {
	var mae float64
	for i := range trueValues {
		mae += math.Abs(predictValues[i] - trueValues[i])
	}
	return mae / float64(len(trueValues))
}
