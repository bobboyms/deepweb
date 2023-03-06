package layers

import "gonum.org/v1/gonum/mat"

type Layer interface {
	AdjustWeights(weight float64)
	Process(input *mat.Dense) *mat.Dense
}
