package mtx

import (
	"gonum.org/v1/gonum/mat"
)

func Add(dataClass [][][]float64) [][]float64 {

	sizeClass := len(dataClass)
	layerMap := make(map[int][][]float64)

	for class := 0; class < sizeClass; class++ {
		dataLayer := dataClass[class]
		for i, layer := range dataLayer {
			v, ok := layerMap[i]
			if ok {
				v = append(v, layer)
				layerMap[i] = v
			} else {
				var v [][]float64
				v = append(v, layer)
				layerMap[i] = v
			}
		}
	}

	add := func(values [][]float64) []float64 {

		zeroDense := CreateDenseWithZeros(1, len(values[0]))
		for _, value := range values {
			sumDense := mat.NewDense(1, len(value), value)
			add := &mat.Dense{}
			add.Add(zeroDense, sumDense)
			zeroDense = add
		}

		return zeroDense.RawMatrix().Data

	}

	sizeMap := len(layerMap)
	newLayers := make([][]float64, sizeMap)
	for i := 0; i < sizeMap; i++ {
		newLayers[i] = add(layerMap[i])
	}

	return newLayers
}

func CreateDenseWithValue(row, col int, value float64) *mat.Dense {
	size := row * col
	values := make([]float64, size)
	for i := 0; i < size; i++ {
		values[i] = value
	}

	return mat.NewDense(row, col, values)
}

func CreateDenseWithZeros(row, col int) *mat.Dense {

	size := row * col
	zeros := make([]float64, size)
	for i := 0; i < size; i++ {
		zeros[i] = 0
	}

	return mat.NewDense(row, col, zeros)

}

func CreateDenseWithNil(row, col int) *mat.Dense {

	return mat.NewDense(row, col, nil)

}

func DenseToSlice(matrix *mat.Dense) [][]float64 {

	row, _ := matrix.Dims()
	values := make([][]float64, row)
	for i := 0; i < row; i++ {
		values[i] = matrix.RowView(i).(*mat.VecDense).RawVector().Data
	}
	return values
}

func SliceToDense(data [][]float64) *mat.Dense {
	rows, cols := len(data), len(data[0])
	flat := make([]float64, rows*cols)
	for i, row := range data {
		for j, val := range row {
			flat[i*cols+j] = val
		}
	}
	return mat.NewDense(rows, cols, flat)
}

func SumCol(values [][]float64) []float64 {
	var results []float64
	for row := 0; row < len(values); row++ {
		var sum float64
		for col := 0; col < len(values[row]); col++ {
			sum += values[row][col]
		}
		results = append(results, sum)
	}
	return results
}

func TransposeToDense(transpose mat.Matrix) *mat.Dense {

	row, col := transpose.Dims()
	var values []float64
	for i := 0; i < row; i++ {
		for j := 0; j < col; j++ {
			values = append(values, transpose.At(i, j))
		}
	}
	return mat.NewDense(row, col, values)
}

func DotProduct(a, b *mat.Dense) float64 {
	row, col := a.Dims()
	size := row * col

	vec1 := mat.NewVecDense(size, a.RawMatrix().Data)
	vec2 := mat.NewVecDense(size, b.RawMatrix().Data)

	return mat.Dot(vec1, vec2)
}
