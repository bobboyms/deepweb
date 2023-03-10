package mtx

import (
	"gonum.org/v1/gonum/mat"
)

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
