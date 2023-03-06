package models

import (
	"deepgo/layers"
	"fmt"
	"gonum.org/v1/gonum/mat"
)

type Sequential struct {
	Layers []layers.Layer
}

func NewSequential() *Sequential {
	return &Sequential{
		Layers: make([]layers.Layer, 0),
	}
}

func (s *Sequential) AddLayer(layer layers.Layer) {
	s.Layers = append(s.Layers, layer)
}

func (s *Sequential) Compile(input *mat.Dense) {

	layer := s.Layers[0]
	output := layer.Process(input)
	for _, layer := range s.Layers[1:] {
		output = layer.Process(output)
	}

	fmt.Printf("\n%v\n", mat.Formatted(output))

}
