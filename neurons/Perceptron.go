package neurons

import (
	"deepgo/stepfunction"
	"fmt"
	"gonum.org/v1/gonum/mat"
)

type Perceptron struct {
	Bias         float64
	Weights      *mat.Dense
	StepFunction stepfunction.Activation
}

func NewPerceptron(weights *mat.Dense, activation stepfunction.Activation, bias float64) Neuron {
	return &Perceptron{
		Bias:         bias,
		Weights:      weights,
		StepFunction: activation,
	}
}

func (p *Perceptron) PrintWeights() {
	fmt.Printf("\n%v\n", mat.Formatted(p.Weights))
}

func (p *Perceptron) Process(input *mat.Dense) float64 {
	dotProduct := DotProduct(input, p.Weights)
	sum := dotProduct + p.Bias
	return p.StepFunction.Activation(sum)
}

func (p *Perceptron) AdjustWeights(weight float64) {
	p.Bias += weight
	row, col := p.Weights.Dims()
	size := row * col
	values := make([]float64, size)
	for i := 0; i < size; i++ {
		values[i] = weight
	}

	weightDense := mat.NewDense(row, col, values)
	adjust := mat.NewDense(row, col, nil)
	adjust.Add(p.Weights, weightDense)

	p.Weights = adjust
}

func DotProduct(input, weights *mat.Dense) float64 {
	row, col := input.Dims()
	size := row * col

	vec1 := mat.NewVecDense(size, input.RawMatrix().Data)
	vec2 := mat.NewVecDense(size, weights.RawMatrix().Data)

	return mat.Dot(vec1, vec2)
}
