package rdkit

/*
#cgo CFLAGS: -I${SRCDIR}/third_party/rdkit
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/third_party/rdkit -lrdkitcffi_linux_amd64 -lm -lstdc++ -lfreetype
#include <stdlib.h>
#include "third_party/rdkit/cffiwrapper.h"
static inline unsigned char* wrap_get_morgan_fp_as_bytes(
    const char* pkl, size_t pkl_sz,
    unsigned long* out_nbytes,
    const char* details_json)
{
    return (unsigned char*)get_morgan_fp_as_bytes(
        pkl, pkl_sz, (size_t*)out_nbytes, details_json
    );
}
*/
import "C"
import (
	"encoding/hex"
	"errors"
	"fmt"
	"unsafe"
)

type Fingerprint struct {
	Bits  []byte
	NBits int
}

func (m *Mol) Morgan(radius, nBits uint, useChirality, useFeatures bool) (Fingerprint, error) {
	if m == nil || m.isDeleted {
		return Fingerprint{}, ErrDeleted
	}
	if nBits == 0 {
		return Fingerprint{}, errors.New("rdkit: nBits must be > 0")
	}
	details := fmt.Sprintf(`{"radius":%d,"nBits":%d,"useChirality":%t,"useFeatures":%t}`, radius, nBits, useChirality, useFeatures)
	cdetails := C.CString(details)
	defer C.free(unsafe.Pointer(cdetails))

	var outNBytes C.ulong
	buf := C.wrap_get_morgan_fp_as_bytes(
		m.pkl, m.pklSize,
		(*C.ulong)(unsafe.Pointer(&outNBytes)),
		cdetails,
	)
	if buf == nil || outNBytes == 0 {
		return Fingerprint{}, errors.New("rdkit: get_morgan_fp_as_bytes failed")
	}
	defer C.free(unsafe.Pointer(buf))

	goBytes := C.GoBytes(unsafe.Pointer(buf), C.int(outNBytes))
	return Fingerprint{Bits: goBytes, NBits: int(nBits)}, nil
}

func (m *Mol) MorganHex(radius, nBits uint, useChirality, useFeatures bool) (string, error) {
	fp, err := m.Morgan(radius, nBits, useChirality, useFeatures)
	if err != nil {
		return "", err
	}
	return hex.EncodeToString(fp.Bits), nil
}
