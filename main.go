package main

import (
	"context"
	"fmt"
	"log"
)

// Record represents a single entry in our temporary in-memory database
type Record struct {
	ID   string
	Text string
	Vec  Vector
}

func main() {
	client := NewOllamaClient("http://localhost:11434", "nomic-embed-text")
	ctx := context.Background() // For simplicity in this test

	// 1. The Data we want to store in our VectorDB
	databaseTexts := []string{
		"LSM trees are optimized for heavy write workloads by appending to a WAL.",
		"The sky appears blue because of Rayleigh scattering in the atmosphere.",
		"Golang handles concurrency efficiently using lightweight goroutines.",
	}

	var database []Record

	// 2. Ingest: Convert text to vectors and store them
	for i, text := range databaseTexts {
		vec, err := client.Embed(ctx, text)
		if err != nil {
			log.Fatalf("Failed to embed: %v", err)
		}
		database = append(database, Record{
			ID:   fmt.Sprintf("doc_%d", i+1),
			Text: text,
			Vec:  vec,
		})
	}

	// 3. The User Query
	searchQuery := "How does Go manage parallel tasks?"
	fmt.Printf("Query: %q\n\n", searchQuery)

	queryVec, err := client.Embed(ctx, searchQuery)
	if err != nil {
		log.Fatalf("Failed to embed query: %v", err)
	}

	// 4. The Brute Force Search (Linear Scan)
	bestScore := float32(-1.0)
	var bestMatch Record

	for _, record := range database {
		// Calculate the distance between the query and the stored record
		score := CosineSimilarity(queryVec, record.Vec)
		fmt.Printf("Score: %.4f | Text: %q\n", score, record.Text)

		if score > bestScore {
			bestScore = score
			bestMatch = record
		}
	}
	
	fmt.Printf("🏆 BEST MATCH (Score: %.4f)\n", bestScore)
	fmt.Printf("=> %s\n", bestMatch.Text)
}
