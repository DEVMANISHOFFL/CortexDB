package main

import (
	"encoding/gob"
	"fmt"
	"os"
)

func (h *HNSW) Save(filename string) error {
	file, err := os.Create(filename)

	if err != nil {
		return fmt.Errorf("failed to create graph: %w", err)
	}

	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(h); err != nil {
		return fmt.Errorf("failed to encode graph: %w", err)
	}

	return nil
}

// LoadHNSW reads a binary file from disk and instantly reconstructs the HNSW graph in memory.
func LoadHNSW(filename string) (*HNSW, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open graph file: %w", err)
	}

	defer file.Close()

	var h HNSW

	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&h); err != nil {
		return nil, fmt.Errorf("failed to decode graph: %w", err)
	}
	return &h, nil
}

