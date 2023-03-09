package neurons

type Perceptron struct {
	Weight float64
}

func NewPerceptron(weight float64) Neuron {
	return &Perceptron{
		Weight: weight,
	}
}

func (p *Perceptron) GetWeight() float64 {
	return p.Weight
}

//func (p *Perceptron) PrintWeights() {
//	fmt.Printf("\n%v\n", mat.Formatted(p.Weights))
//}

func (p *Perceptron) Process(input float64) float64 {
	return input * p.Weight
	//dotProduct := mtx.DotProduct(input, p.Weights)
	//sum := dotProduct //+ p.Bias
	//return p.StepFunction.Activation(sum)
}

func (p *Perceptron) AdjustWeights(weight float64) {
	p.Weight = weight
}
