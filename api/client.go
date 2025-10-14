// MIT License
// Copyright (c) 2025
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to do so, subject to the
// following conditions:
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
	"context"
	"crypto/sha1"
	"encoding/binary"
	"errors"
	"log"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

const DimBits = 1024

type API struct {
	client client.Client
}

func MustConnect(addr string) *API {
	cli, err := client.NewClient(context.Background(), client.Config{Address: addr})
	if err != nil {
		panic(err)
	}
	return &API{client: cli}
}

func (a *API) Close() { a.client.Close() }


func (a *API) EnsureCollection(ctx context.Context, name string) error {
	has, err := a.client.HasCollection(ctx, name)
	if err != nil {
		return err
	}
	if has {
		return nil
	}
	schema := &entity.Schema{
		CollectionName: name,
		AutoID:         true,
		Fields: []*entity.Field{
			{Name: "id", DataType: entity.FieldTypeInt64, AutoID: true, PrimaryKey: true},
			{Name: "smiles", DataType: entity.FieldTypeVarChar, TypeParams: map[string]string{"max_length": "512"}},
			{Name: "vec", DataType: entity.FieldTypeBinaryVector, TypeParams: map[string]string{"dim": "1024"}},
		},
	}
	return a.client.CreateCollection(ctx, schema, entity.DefaultShardNumber)
}

func (a *API) BuildIndex(ctx context.Context, coll string, nlist int) error {
	if nlist <= 0 {
		nlist = 4096
	}
	idx, err := entity.NewIndexBinIvfFlat(entity.JACCARD, nlist)
	if err != nil {
		return err
	}
	if err := a.client.CreateIndex(ctx, coll, "vec", idx, false); err != nil {
		return err
	}
	return a.client.LoadCollection(ctx, coll, false)
}

func (a *API) Load(ctx context.Context, coll string) error    { return a.client.LoadCollection(ctx, coll, false) }
func (a *API) Release(ctx context.Context, coll string) error { return a.client.ReleaseCollection(ctx, coll) }

func (a *API) Flush(ctx context.Context, coll string) error { return a.client.Flush(ctx, coll, false) }
func (a *API) DropCollection(ctx context.Context, coll string) error {
	return a.client.DropCollection(ctx, coll)
}

func (a *API) BitsFromSmiles(smiles []string) [][]byte {
	out := make([][]byte, len(smiles))
	for i, s := range smiles {
		out[i] = HashBits1024(s)
	}
	return out
}

func (a *API) InsertBits(ctx context.Context, coll string, smiles []string, bits [][]byte, doFlush bool) error {
	smCol := entity.NewColumnVarChar("smiles", smiles)
	vecCol := entity.NewColumnBinaryVector("vec", DimBits, bits)
	if _, err := a.client.Insert(ctx, coll, "", smCol, vecCol); err != nil {
		return err
	}
	log.Printf("insert batch collection=%s rows=%d", coll, len(smiles))
	if doFlush {
		if err := a.client.Flush(ctx, coll, false); err != nil {
			return err
		}
		log.Printf("insert flush collection=%s rows=%d", coll, len(smiles))
	}
	return nil
}

func (a *API) SearchTop1Bits(ctx context.Context, coll string, queries [][]byte) ([]int64, []float32, []string, error) {
	vecs := make([]entity.Vector, 0, len(queries))
	for _, q := range queries {
		vecs = append(vecs, entity.BinaryVector(q))
	}
	sp, err := entity.NewIndexBinIvfFlatSearchParam(64)
	if err != nil {
		return nil, nil, nil, err
	}
	res, err := a.client.Search(ctx, coll, nil, "", []string{"smiles"}, vecs, "vec", entity.JACCARD, 1, sp)
	if err != nil {
		return nil, nil, nil, err
	}
	var ids []int64
	var scores []float32
	var matches []string
	for _, sr := range res {
		if sr.Err != nil {
			return nil, nil, nil, sr.Err
		}
		smilesCol := sr.Fields.GetColumn("smiles")
		if smilesCol == nil {
			return nil, nil, nil, errors.New("missing smiles")
		}
		for i := 0; i < sr.ResultCount; i++ {
			id, _ := sr.IDs.GetAsInt64(i)
			sm, _ := smilesCol.GetAsString(i)
			d := sr.Scores[i]
			s := float32(1.0) - d
			ids = append(ids, id)
			matches = append(matches, sm)
			scores = append(scores, s)
		}
	}
	return ids, scores, matches, nil
}

func (a *API) CollectionRowCount(ctx context.Context, coll string) (int64, error) {
	stats, err := a.client.GetCollectionStatistics(ctx, coll)
	if err != nil {
		return 0, err
	}
	val, ok := stats["row_count"]
	if !ok {
		return 0, errors.New("row_count not found in stats")
	}
	var cnt int64
	for i := 0; i < len(val); i++ {
		c := val[i]
		if c < '0' || c > '9' {
			continue
		}
		cnt = cnt*10 + int64(c-'0')
	}
	return cnt, nil
}

func HashBits1024(s string) []byte {
	h := sha1.Sum([]byte(s))
	out := make([]byte, DimBits/8)
	for i := 0; i < len(out); i++ {
		v := binary.BigEndian.Uint16([]byte{h[i%len(h)], h[(i+1)%len(h)]})
		out[i] = byte(v & 0xFF)
	}
	return out
}
