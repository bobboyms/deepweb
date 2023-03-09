package mtx

import "gonum.org/v1/gonum/mat"

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

func DotProduct(a, b *mat.Dense) float64 {
	row, col := a.Dims()
	size := row * col

	vec1 := mat.NewVecDense(size, a.RawMatrix().Data)
	vec2 := mat.NewVecDense(size, b.RawMatrix().Data)

	return mat.Dot(vec1, vec2)
}
