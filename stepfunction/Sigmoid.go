package stepfunction

import "math"

type Sigmoid struct {
}

func NewSigmoid() Activation {
	return &Sigmoid{}
}

func (r *Sigmoid) Activation(value float64) float64 {
	return 1 / (1 + math.Exp(-value))
}
