package stepfunction

import "gonum.org/v1/gonum/mat"

type Activation interface {
	Activation(data *mat.Dense) *mat.Dense
}
