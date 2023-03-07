package layers

import (
	"deepgo/neurons"
	"deepgo/stepfunction"
	"gonum.org/v1/gonum/mat"
	"math/rand"
	"sync"
	"time"
)

type Dense struct {
	Size         int
	Input        *mat.Dense
	Neurons      Dims
	InputDims    Dims
	Perceptrons  []neurons.Neuron
	StepFunction stepfunction.Activation
}

func NewDense(inputDims Dims, neurons Dims, stepFunction stepfunction.Activation) Layer {

	bias := 0.1
	size := neurons.Rows * neurons.Cols

	return &Dense{
		InputDims:    inputDims,
		Neurons:      neurons,
		Size:         size,
		StepFunction: stepFunction,
		Perceptrons:  CreatePerceptrons(inputDims, size, stepFunction, bias),
	}

}

func (d *Dense) AdjustWeights(weight float64) {

	var wg sync.WaitGroup
	wg.Add(d.Size)
	for _, p := range d.Perceptrons {
		perceptron := p
		go func() {
			defer wg.Done()
			perceptron.AdjustWeights(weight)
		}()
	}
	wg.Wait()

}

func (d *Dense) Process(input *mat.Dense) *mat.Dense {

	values := make([]float64, d.Size)

	var wg sync.WaitGroup
	wg.Add(d.Size)
	for i, p := range d.Perceptrons {
		i := i
		perceptron := p
		go func() {
			defer wg.Done()
			values[i] = perceptron.Process(input)
		}()
	}
	wg.Wait()

	if _, ok := d.StepFunction.(*stepfunction.Softmax); ok {
		return mat.NewDense(
			d.Neurons.Rows,
			d.Neurons.Cols,
			stepfunction.SoftmaxFunction(values))
	}

	return mat.NewDense(d.Neurons.Rows, d.Neurons.Cols, values)

}

func CreatePerceptrons(inputDims Dims, size int, activation stepfunction.Activation, bias float64) []neurons.Neuron {

	perceptrons := make([]neurons.Neuron, size)
	for i := 0; i < size; i++ {
		weights := CreateWeights(inputDims.Rows, inputDims.Cols, 10)
		perceptrons[i] = neurons.NewPerceptron(weights, activation, bias)
	}

	return perceptrons
}

func CreateWeights(row, col int, div float64) *mat.Dense {
	size := row * col
	values := make([]float64, size)
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < size; i++ {
		if div > 0 {
			values[i] = rand.Float64() / div
		} else {
			values[i] = rand.Float64()
		}

	}
	return mat.NewDense(row, col, values)
}
