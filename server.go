package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

type APIServer struct {
	DB           *DB
	OllamaClient *OllamaClient
}

type InsertRequest struct {
	ID   string `json:"id"`
	Text string `json:"text"`
}

type SearchRequest struct {
	Query string `json:"query"`
}

type SearchResponse struct {
	ID    string  `json:"id"`
	Text  string  `json:"text"`
	Score float32 `json:"score"`
}

func (s *APIServer) handleInsert(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req InsertRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON body", http.StatusBadRequest)
		return
	}

	// get the embedding from Ollama
	ctx := context.Background()
	vec, err := s.OllamaClient.Embed(ctx, req.Text)
	if err != nil {
		http.Error(w, "Failed to generate embedding", http.StatusInternalServerError)
		return
	}

	// Insert into the Database (WAL, Memtable, and HNSW Graph)
	if err := s.DB.Insert(req.ID, req.Text, vec); err != nil {
		http.Error(w, "Database insertion failed", http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, `{"status": "success", "message": "Inserted %s"}`, req.ID)
}

func (s *APIServer) handleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON body", http.StatusBadRequest)
		return
	}

	ctx := context.Background()
	queryVec, err := s.OllamaClient.Embed(ctx, req.Query)
	if err != nil {
		http.Error(w, "Failed to generate query embedding", http.StatusInternalServerError)
		return
	}

	record, score := s.DB.Search(queryVec)

	resp := SearchResponse{
		ID:    record.ID,
		Text:  record.Text,
		Score: score,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *APIServer) Start(port string) error {
	mux := http.NewServeMux()
	mux.HandleFunc("/insert", s.handleInsert)
	mux.HandleFunc("/search", s.handleSearch)

	fmt.Printf("Vector Database API running on http://localhost:%s\n", port)
	return http.ListenAndServe(":"+port, mux)
}
