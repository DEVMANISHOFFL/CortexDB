package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

// it will store 768 numbers, so if i use float64 it will take 2x memory(ram) uselessly.thats why i used float32
type Vector []float32
	
type Embedder interface {
	Embed(ctx context.Context, text string) (Vector, error)
}

type OllamaClient struct {
	BaseURL    string
	Model      string
	HTTPClient *http.Client
}

func NewOllamaClient(BaseURL, model string) *OllamaClient {
	return &OllamaClient{
		BaseURL:    BaseURL,
		Model:      model,
		HTTPClient: &http.Client{},
	}
}

type OllamaRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

type OllamaResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

func (c *OllamaClient) Embed(ctx context.Context, text string) (Vector, error) {
	reqBody := OllamaRequest{
		Model: c.Model,
		Input: text,
	}

	jsonBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	endPoint := fmt.Sprintf("%s/api/embed", c.BaseURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endPoint, bytes.NewBuffer(jsonBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama API call failed: %w", err)
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama returned non-200 status: %d", resp.StatusCode)
	}

	var ollamaResp OllamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(ollamaResp.Embeddings) == 0 || len(ollamaResp.Embeddings[0]) == 0 {
		return nil, fmt.Errorf("ollama returned an empty embedding array")
	}

	return ollamaResp.Embeddings[0], nil
}
