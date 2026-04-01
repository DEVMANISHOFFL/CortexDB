package main

import (
	"encoding/binary"
	"math"
)

// SerializeVector convert a slice of floats into a dense byte array using bit-level casting.
func SerializeVector(vec Vector) []byte {
	// a. preallocate the exact siize
	// 1 float32 = 4 bytes. So we need len(vev) * 4 bytes.
	// by specifying the exact capacity, Go allocates this one and never resize it.
	b := make([]byte, len(vec)*4)

	// b. loop through without reflection
	for i, f := range vec {
		// math.Float32bits takes the raw 1s and 0s of the float and pretends they are an integer.
		// This is just a type-cast at the CPU level, it takes almost 0 time.
		bits := math.Float32bits(f)

		// Write the 4 bytes directly into our exact offset in the slice
		binary.LittleEndian.PutUint32(b[i*4:], bits)
	}
	return b
}

func DeserializeVector(data []byte) Vector {
	if len(data)%4 != 0 {
		return nil
	}

	vec := make(Vector, len(data)/4)

	for i := range vec {
		bits := binary.LittleEndian.Uint32(data[i*4:])

		vec[i] = math.Float32frombits(bits)
	}
	return vec
}
