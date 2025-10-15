// GNU GENERAL PUBLIC LICENSE
// Copyright (c) 2025
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"
)

type InsertJSON struct {
	Collection string   `json:"collection"`
	Smiles     []string `json:"smiles"`
}

type SearchReq struct {
	Collection string   `json:"collection"`
	Smiles     []string `json:"smiles"`
}

type SearchResult struct {
	Input  string  `json:"input"`
	Match  string  `json:"match"`
	Score  float32 `json:"score"`
}

type SearchResp struct {
	Results []SearchResult `json:"results"`
}

type InfoResp struct {
	Collection     string `json:"collection"`
	RowCount       int64  `json:"row_count"`
	DimBits        int    `json:"dim_bits"`
	EstimatedBytes int64  `json:"estimated_bytes"`
}

func envOr(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

func main() {
	addr := envOr("MILVUS_ADDR", "localhost:19530")
	api := MustConnect(addr)
	defer api.Close()
	log.Printf("boot cores=%d milvus=%s", runtime.NumCPU(), addr)

	http.HandleFunc("/insert", func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		ct := r.Header.Get("Content-Type")

		if strings.HasPrefix(ct, "application/json") {
			var req InsertJSON
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, err.Error(), 400)
				return
			}
			if req.Collection == "" {
				http.Error(w, "missing collection", 400)
				return
			}
			log.Printf("insert(json) recv collection=%s n=%d", req.Collection, len(req.Smiles))
			ctx, cancel := context.WithTimeout(r.Context(), 2*time.Hour)
			defer cancel()
			if err := api.EnsureCollection(ctx, req.Collection); err != nil {
				http.Error(w, "collection: "+err.Error(), 500)
				return
			}
			t0 := time.Now()
			bits := api.BitsFromSmiles(req.Smiles)
			log.Printf("insert(json) vectors_done collection=%s n=%d took_ms=%d", req.Collection, len(bits), time.Since(t0).Milliseconds())
			if err := api.InsertBits(ctx, req.Collection, req.Smiles, bits, true); err != nil {
				http.Error(w, "insert: "+err.Error(), 500)
				return
			}
			log.Printf("insert(json) success collection=%s n=%d total_ms=%d", req.Collection, len(req.Smiles), time.Since(start).Milliseconds())
			w.WriteHeader(204)
			return
		}

		coll := r.URL.Query().Get("collection")
		if coll == "" {
			http.Error(w, "missing collection", 400)
			return
		}
		batch := 10000
		if v := r.URL.Query().Get("batch"); v != "" {
			if n, err := strconv.Atoi(v); err == nil && n > 0 {
				batch = n
			}
		}
		flushEvery := 20
		if v := r.URL.Query().Get("flush_every"); v != "" {
			if n, err := strconv.Atoi(v); err == nil && n > 0 {
				flushEvery = n
			}
		}

		log.Printf("insert(stream) recv collection=%s batch=%d flush_every=%d", coll, batch, flushEvery)
		ctx, cancel := context.WithTimeout(r.Context(), 24*time.Hour)
		defer cancel()
		if err := api.EnsureCollection(ctx, coll); err != nil {
			http.Error(w, "collection: "+err.Error(), 500)
			return
		}

		sc := bufio.NewScanner(r.Body)
		buf := make([]byte, 0, 1024*1024)
		sc.Buffer(buf, 16*1024*1024)

		smBatch := make([]string, 0, batch)
		btBatch := make([][]byte, 0, batch)
		var total int64
		batches := 0

		insertBatch := func() error {
			if len(smBatch) == 0 {
				return nil
			}
			if err := api.InsertBits(ctx, coll, smBatch, btBatch, false); err != nil {
				return err
			}
			total += int64(len(smBatch))
			batches++
			log.Printf("insert(stream) batch collection=%s rows=%d total=%d", coll, len(smBatch), total)
			smBatch = smBatch[:0]
			btBatch = btBatch[:0]
			if batches%flushEvery == 0 {
				if err := api.Flush(ctx, coll); err != nil {
					return err
				}
				log.Printf("insert(stream) flush checkpoint collection=%s total=%d", coll, total)
			}
			return nil
		}

		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if line == "" {
				continue
			}
			smBatch = append(smBatch, line)
			btBatch = append(btBatch, api.BitsFromSmile(line))
			if len(smBatch) >= batch {
				if err := insertBatch(); err != nil {
					http.Error(w, "insert: "+err.Error(), 500)
					return
				}
			}
		}
		if err := sc.Err(); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		if err := insertBatch(); err != nil {
			http.Error(w, "insert: "+err.Error(), 500)
			return
		}
		if err := api.Flush(ctx, coll); err != nil {
			http.Error(w, "flush: "+err.Error(), 500)
			return
		}
		log.Printf("insert(stream) success collection=%s total_rows=%d total_ms=%d", coll, total, time.Since(start).Milliseconds())
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"collection":  coll,
			"inserted":    total,
			"batch":       batch,
			"flush_every": flushEvery,
			"ms":          time.Since(start).Milliseconds(),
		})
	})

	http.HandleFunc("/search", func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		var req SearchReq
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		if req.Collection == "" {
			http.Error(w, "missing collection", 400)
			return
		}
		log.Printf("search recv collection=%s n=%d", req.Collection, len(req.Smiles))
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
		defer cancel()
		bits := api.BitsFromSmiles(req.Smiles)
		_, scores, matches, err := api.SearchTop1Bits(ctx, req.Collection, bits)
		if err != nil {
			http.Error(w, "search: "+err.Error(), 500)
			return
		}
		out := make([]SearchResult, len(req.Smiles))
		for i := range req.Smiles {
			out[i] = SearchResult{Input: req.Smiles[i], Match: matches[i], Score: scores[i]}
		}
		log.Printf("search success collection=%s n=%d total_ms=%d", req.Collection, len(req.Smiles), time.Since(start).Milliseconds())
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(SearchResp{Results: out})
	})

	http.HandleFunc("/info", func(w http.ResponseWriter, r *http.Request) {
		coll := r.URL.Query().Get("collection")
		if coll == "" {
			http.Error(w, "missing collection", 400)
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
		defer cancel()
		rows, err := api.CollectionRowCount(ctx, coll)
		if err != nil {
			http.Error(w, "info: "+err.Error(), 500)
			return
		}
		est := rows * int64(DimBits/8)
		resp := InfoResp{Collection: coll, RowCount: rows, DimBits: DimBits, EstimatedBytes: est}
		log.Printf("info collection=%s row_count=%d dim_bits=%d est_bytes=%d", coll, rows, DimBits, est)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})

	http.HandleFunc("/reset", func(w http.ResponseWriter, r *http.Request) {
		coll := r.URL.Query().Get("collection")
		if coll == "" {
			http.Error(w, "missing collection", 400)
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
		defer cancel()
		if err := api.DropCollection(ctx, coll); err != nil {
			http.Error(w, "drop: "+err.Error(), 500)
			return
		}
		if err := api.EnsureCollection(ctx, coll); err != nil {
			http.Error(w, "recreate: "+err.Error(), 500)
			return
		}
		log.Printf("reset collection=%s", coll)
		w.WriteHeader(204)
	})

	log.Println("listen :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
