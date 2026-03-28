package main

import (
	"context"
	"fmt"
	"log"
	"time"
)

func main() {
	client := NewOllamaClient("http://localhost:11434", "nomic-embed-text")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	text := "Hey there! I'm Manish, a Golang Developer"
	fmt.Printf("Generating embedding for: %q\n", text)
	start := time.Now()

	vector, err := client.Embed(ctx, text)
	if err != nil {
		log.Fatalf("Fatal: Failed to generate embedding:%v", err)
	}

	fmt.Printf("Success, (took %v)\n", time.Since(start))
	fmt.Printf("Vector Length: %d dimensions\n", len(vector))

	if len(vector) > 5 {
		fmt.Printf("First 5 floats: %v...\n", vector[:5])
	}
}
