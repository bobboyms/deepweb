package stepfunction

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

type Sigmoid struct {
}

func NewSigmoid() Activation {
	return &Sigmoid{}
}

func (s *Sigmoid) Activation(data *mat.Dense) *mat.Dense {
	r, c := data.Dims()
	results := make([]float64, r*c)
	for i, value := range data.RawMatrix().Data {
		results[i] = 1.0 / (1.0 + math.Exp(-value))
	}

	return mat.NewDense(r, c, results)
}
