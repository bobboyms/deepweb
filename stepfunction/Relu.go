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