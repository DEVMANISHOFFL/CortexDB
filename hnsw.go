package main

import (
	"math"
	"math/rand"
	"sort"
)

type Node struct {
	ID        string
	Vector    Vector
	MaxLayer  int
	Neighbors [][]string
}

type HNSW struct {
	Nodes      map[string]*Node
	EntryPoint string
	MaxLayer   int

	// hyperparametres
	M              int     // max neighbours per layer
	M0             int     // max neighbors at Layer 0 (the dense base layer)
	EfConstruction int     // size of the dynamic candidate pool during insertion
	LevelMult      float64 // multiplier for the probability distribution
}

func NewHNSW(m int, efConstruction int) *HNSW {
	return &HNSW{
		Nodes:          make(map[string]*Node),
		EntryPoint:     "",
		MaxLayer:       -1, // means the graoh is completely empty
		M:              m,
		M0:             m * 2,
		EfConstruction: efConstruction,
		// this multiplier ensures levels decay exponentially
		LevelMult: 1.0 / math.Log(float64(m)),
	}
}

func (h *HNSW) randomLevel() int {
	f := rand.Float64()

	if f == 0.0 {
		f = 0.0000001
	}

	level := int(-math.Log(f) * h.LevelMult)
	return level
}

// searchLayer greedily navigates a specific layer to find the closest node to the query
// it returns the ID of the closest node found on this layer.
func (h *HNSW) searchLayer(query Vector, entryPointID string, layer int) string {
	if entryPointID == "" {
		return ""
	}

	currentBestNodeID := entryPointID
	currentBestNode := h.Nodes[entryPointID]

	currentBestScore := CosineSimilarity(query, currentBestNode.Vector)

	for {
		changed := false
		currentNode := h.Nodes[currentBestNodeID]

		if layer >= len(currentNode.Neighbors) {
			break
		}

		// evaluate all friends at this specific altitude
		for _, neighborID := range currentNode.Neighbors[layer] {
			neighborNode := h.Nodes[neighborID]
			score := CosineSimilarity(query, neighborNode.Vector)

			if score > currentBestScore {
				currentBestScore = score
				currentBestNodeID = neighborID
				changed = true
			}
		}

		// if looped through all neighbors and didn't update 'changed' to true,
		// it means no neighbor is closer than where we currently stand.
		// We have hit the dead end for this layer, so we break the loop.
		if !changed {
			break
		}
	}

	// Return the ID of the node where this finally stopped
	return currentBestNodeID
}

func (h *HNSW) Insert(id string, vector Vector) {
	level := h.randomLevel()

	newNode := &Node{
		ID:        id,
		Vector:    vector,
		MaxLayer:  level,
		Neighbors: make([][]string, level+1),
	}

	h.Nodes[id] = newNode

	if h.EntryPoint == "" {
		h.EntryPoint = id
		h.MaxLayer = level
		return
	}

	currentBestNodeID := h.EntryPoint

	// If the graph is taller than our new node, route down to our starting layer.
	for layer := h.MaxLayer; layer > level; layer-- {
		currentBestNodeID = h.searchLayer(vector, currentBestNodeID, layer)
	}

	//(Search and connect on every layer down to 0)
	for layer := min(level, h.MaxLayer); layer >= 0; layer-- {
		currentBestNodeID = h.searchLayer(vector, currentBestNodeID, layer)

		maxM := h.M
		if layer == 0 {
			maxM = h.M0
		}

		h.wireNeighbors(newNode, currentBestNodeID, layer, maxM)
	}
	//  Update Entry Point if this new node is the highest one we've ever seen
	if level > h.MaxLayer {
		h.MaxLayer = level
		h.EntryPoint = id
	}
}

// wireNeighbors connects the newNode to the closestNode and ensures bidirectional links.
func (h *HNSW) wireNeighbors(newNode *Node, closestNodeID string, layer int, maxM int) {
	closestNode := h.Nodes[closestNodeID]

	// Link A to B
	newNode.Neighbors[layer] = append(newNode.Neighbors[layer], closestNodeID)
	// Link B to A
	closestNode.Neighbors[layer] = append(closestNode.Neighbors[layer], newNode.ID)

	if len(closestNode.Neighbors[layer]) > maxM {
		h.pruneNeighbors(closestNode, layer, maxM)
	}
}

// pruneNeighbors removes the weakest connection if a node has too many friends.
func (h *HNSW) pruneNeighbors(node *Node, layer int, maxM int) {
	type candidate struct {
		id    string
		score float32
	}

	var friends []candidate
	for _, neighborID := range node.Neighbors[layer] {
		neighborNode := h.Nodes[neighborID]
		score := CosineSimilarity(node.Vector, neighborNode.Vector)
		friends = append(friends, candidate{id: neighborID, score: score})
	}

	// Sort by score descending (highest similarity first)
	sort.Slice(friends, func(i, j int) bool {
		return friends[i].score > friends[j].score
	})

	// Keep only the top maxM friends
	node.Neighbors[layer] = make([]string, 0, maxM)
	for i := range maxM {
		node.Neighbors[layer] = append(node.Neighbors[layer], friends[i].id)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
