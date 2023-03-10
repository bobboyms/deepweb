package training

//camadaOcultaTransposta = camadaOculta.T
//pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
//pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)

//func WeightAdjustmentCalculation(nextDelta []float64, activations []float64, oldWeight [][]float64, momentum, learningRate float64) float64 {
//	size := len(activations)
//	deltas := make([]float64, size)
//	sumWeights := mtx.SumCol(oldWeight)
//	for j := 0; j < size; j++ {
//		var sum float64
//		for i := 0; i < len(nextDelta); i++ {
//			//sum += SigmoidDerivative(activation[j]) * sumWeights[j] * outputDeltas[i]
//			sum += (sumWeights[j] * momentum)
//			//(pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)
//		}
//		deltas[j] = sum
//	}
//	return deltas
//}
