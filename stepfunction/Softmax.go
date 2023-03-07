package stepfunction

import "math"

type Softmax struct {
}

func NewSoftmax() Activation {
	return &Softmax{}
}

func (s *Softmax) Activation(value float64) float64 {
	return value
}

func SoftmaxFunction(vector []float64) []float64 {
	e := make([]float64, len(vector))
	sum := 0.0
	for i, v := range vector {
		e[i] = math.Exp(v)
		sum += e[i]
	}

	for i := range e {
		e[i] /= sum
	}

	return e
}
