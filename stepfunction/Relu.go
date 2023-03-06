package stepfunction

type Relu struct {
}

func NewRelu() Activation {
	return &Relu{}
}

func (r *Relu) Activation(value float64) float64 {
	if value < 0 {
		return 0
	} else {
		return value
	}
}

/// sigmoid implements the sigmoid function
//// for use in activation functions.
//func sigmoid(x float64) float64 {
//	return 1.0 / (1.0 + math.Exp(-x))
//}
//
//// sigmoidPrime implements the derivative
//// of the sigmoid function for backpropagation.
//func sigmoidPrime(x float64) float64 {
//	return x * (1.0 - x)
//}
