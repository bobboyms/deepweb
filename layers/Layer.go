package layers

import "gonum.org/v1/gonum/mat"

type Layer interface {
	PrintWeights()
	GetWeights() *mat.Dense
	AdjustWeights(weights *mat.Dense)
	Process(input []float64) *mat.Dense
}
