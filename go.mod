module github.com/mdkelley02/streamscribe

go 1.26.0

replace github.com/ggerganov/whisper.cpp/bindings/go => ./whisper.cpp/bindings/go

require (
	github.com/ggerganov/whisper.cpp/bindings/go v0.0.0-20260507042818-c81b2dabbc45
	github.com/stretchr/testify v1.9.0
)

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
