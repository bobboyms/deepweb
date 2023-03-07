package stepfunction

const (
	Sunday = iota
	Monday
	Tuesday
	Wednesday
	Thursday
	Friday
	Saturday
)

type DayOfWeek int

func teste(d DayOfWeek) {

}

func x() {
	teste(Friday)
}

type Activation interface {
	Activation(float642 float64) float64
}
