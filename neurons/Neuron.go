package neurons

import "gonum.org/v1/gonum/mat"

type Neuron interface {
	Process(input *mat.Dense) float64
	AdjustWeights(value float64)
	PrintWeights()
}
