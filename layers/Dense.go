package layers

import (
	"deepgo/mtx"
	"deepgo/stepfunction"
	"fmt"
	"gonum.org/v1/gonum/mat"
)

type Dense struct {
	InSize, OutSize int
	Weights         *mat.Dense
	StepFunction    stepfunction.Activation
}

func NewDense(inSize, outSize int, activation stepfunction.Activation) Layer {

	return &Dense{
		InSize:       inSize,
		OutSize:      outSize,
		StepFunction: activation,
	}
}

func (d *Dense) PrintWeights() {
	fmt.Printf("\n%v\n", mat.Formatted(d.Weights))
}

func (d *Dense) AdjustWeights(weights *mat.Dense) {
	d.Weights = weights
}

func (d *Dense) Process(input []float64) *mat.Dense {
	_, wCol := d.Weights.Dims()
	sumDense := mtx.CreateDenseWithValue(1, wCol, 0)
	for i, record := range input {
		r := d.Weights.RowView(i).(*mat.VecDense).RawVector().Data

		sum := &mat.Dense{}
		sum.Add(sumDense, mat.NewDense(1, wCol, Multiply(record, r)))
		sumDense = sum
	}

	return d.StepFunction.Activation(sumDense)
}

func (d *Dense) GetWeights() *mat.Dense {
	return d.Weights
}

func Multiply(input float64, weights []float64) []float64 {
	results := make([]float64, len(weights))
	for i, weight := range weights {
		results[i] += input * weight
	}
	return results
}
