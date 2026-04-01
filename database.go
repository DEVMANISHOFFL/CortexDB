package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sync"
)

// db is the master controller unifying storage and vector search.
type DB struct {
	mu      sync.RWMutex
	KVStore map[string]Record // memtable
	Graph   *HNSW             // vector index
	walFile *os.File          // write ahead log
}

func InitDB(graphPath, walPath string) (*DB, error) {
	db := &DB{
		KVStore: make(map[string]Record),
	}

	if _, err := os.Stat(graphPath); err == nil {
		fmt.Println("Loading HNSW Graph from disk...")
		db.Graph, _ = LoadHNSW(graphPath)
	} else {
		fmt.Println("Initializing new HNSW Graph...")
		db.Graph = NewHNSW(16, 100)
	}

	var err error
	db.walFile, err = os.OpenFile(walPath, os.O_APPEND|os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open WAL: %w", err)
	}

	fmt.Println("Replaying Write-Ahead Log...")
	scanner := bufio.NewScanner(db.walFile)
	for scanner.Scan() {
		var rec Record
		if err := json.Unmarshal(scanner.Bytes(), &rec); err == nil {
			db.KVStore[rec.ID] = rec
		}
	}

	fmt.Printf("Database initialized with %d records.\n", len(db.KVStore))
	return db, nil
}

// Insert securely adds data to both the Storage Engine and the Vector Index.
func (db *DB) Insert(id string, text string, vec Vector) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	rec := Record{ID: id, Text: text, Vec: vec}

	logEntry, _ := json.Marshal(rec)
	logEntry = append(logEntry, '\n')
	if _, err := db.walFile.Write(logEntry); err != nil {
		return err
	}

	db.KVStore[id] = rec

	if _, exists := db.Graph.Nodes[id]; !exists {
		db.Graph.Insert(id, vec)
	}

	return nil
}

// Search coordinates the embedding routing and the storage lookup.
func (db *DB) Search(queryVec Vector) (Record, float32) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.Graph.EntryPoint == "" {
		return Record{}, 0.0
	}

	bestMatchID := db.Graph.EntryPoint
	for layer := db.Graph.MaxLayer; layer >= 0; layer-- {
		bestMatchID = db.Graph.searchLayer(queryVec, bestMatchID, layer)
	}

	bestMatchRecord := db.KVStore[bestMatchID]
	score := CosineSimilarity(queryVec, bestMatchRecord.Vec)

	return bestMatchRecord, score
}

// Close safely shuts down the database, saving the graph state.
func (db *DB) Close(graphPath string) {
	db.mu.Lock()
	defer db.mu.Unlock()

	fmt.Println("Saving graph state to disk...")
	db.Graph.Save(graphPath)
	db.walFile.Close()
}
