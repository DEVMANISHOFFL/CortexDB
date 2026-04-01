package main

import (
	"log"
)

// Record represents a single entry in our key-value database
type Record struct {
	ID   string
	Text string
	Vec  Vector
}

func main() {
	client := NewOllamaClient("http://localhost:11434", "nomic-embed-text")

	db, err := InitDB("graph.bin", "wal.log")
	if err != nil {
		log.Fatalf("Failed to initialize database: %v", err)
	}

	defer db.Close("graph.bin")

	server := &APIServer{
		DB:           db,
		OllamaClient: client,
	}

	if err := server.Start("8080"); err != nil {
		log.Fatalf("Server crashed: %v", err)
	}
}
