package neurons

type Neuron interface {
	Process(input float64) float64
	AdjustWeights(weight float64)
	GetWeight() float64
}
