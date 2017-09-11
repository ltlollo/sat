#include "logic_primitives.h"

#include <immintrin.h>
#include <memory.h>
#include <string.h>
#include <assert.h>

#ifndef INTERN
#define INTERN static
#endif

INTERN inline void
vec1_fill(void *ptr, size_t size, __m256i reg) {

	for (size_t i = 0; i < size; i++) {
		_mm256_store_si256(ptr, reg);
		ptr = (char *)ptr + sizeof (__m256i);
	}
}

EXPORT void
vec1_fill_ones(void *ptr, size_t size) {
	vec1_fill(ptr, size, _mm256_set1_epi64x(-1));
}

INTERN inline void
vec1_set_pos(void *ptr, size_t pos, _short_t_ val) {
	size_t pos_dw = pos / (sizeof (uint64_t) * 4);
	uint64_t *dwp = ptr;
	short bit_offs = 2 * (pos & 0x1f);
	uint64_t mask = ~(0b11ull << bit_offs);
	uint64_t val_reg = val << bit_offs;
	dwp[pos_dw] = dwp[pos_dw] & mask | val_reg;
}

EXPORT void
vec1_dill_var_pos(void *ptr, size_t size, size_t pos, _short_t_ var) {
	assert(pos < size * 128);

	vec1_fill_ones(ptr, size);
	vec1_set_pos(ptr, pos, var);
}

INTERN inline _short_t_
vec1_get_pos(void *ptr, size_t pos) {
	size_t pos_dw = pos / (sizeof (uint64_t) * 4);
	uint64_t *dwp = ptr;
	short bit_offs = 2 * (pos & 0x1f);
	return (dwp[pos_dw] >> bit_offs) & 0b11;
}

INTERN inline _short_t_
vec1_or_pos(void *ptr, size_t pos, _short_t_ val) {
	size_t pos_dw = pos / (sizeof (uint64_t) * 4);
	uint64_t *dwp = ptr;
	short bit_offs = 2 * (pos & 0x1f);
	dwp[pos_dw] |= (uint64_t)val << bit_offs;
	return (dwp[pos_dw] >> bit_offs) & 0b11;
}

INTERN inline _short_t_
vec1_and_pos(void *ptr, size_t pos, _short_t_ val) {
	size_t pos_dw = pos / (sizeof (uint64_t) * 4);
	uint64_t *dwp = ptr;
	short bit_offs = 2 * (pos & 0x1f);
	dwp[pos_dw] &= (val | ~0b11ull) << bit_offs;
	return (dwp[pos_dw] >> bit_offs) & 0b11;
}

INTERN inline _short_t_
element_negate(_short_t_ val) {
	return ~val & 0b11;
}

EXPORT void
vec1_copy(void *restrict dst, void *restrict src, size_t size) {
	__m256i src_reg;

	for (size_t i = 0; i < size; i++) {
		src_reg = _mm256_load_si256(src);
		_mm256_stream_si256(dst, src_reg);
		src = (char *)src + sizeof (__m256i);
		dst = (char *)dst + sizeof (__m256i);
	}
}

EXPORT void
vec1_var_pos(void *ptr, size_t size, size_t pos,  _short_t_ val) {
	assert(pos < size * 128);

	vec1_fill_ones(ptr, size);
	vec1_set_pos(ptr, pos, val);
}

EXPORT _short_t_
vec1_all_ones(void *ptr, size_t size) {
	__m256i one_reg = _mm256_set1_epi64x(-1);

	for (size_t i = 0; i < size; i++) {
		__m256i val = _mm256_load_si256(ptr);
		
		if (_mm256_testc_si256(val, one_reg) == 0) {
			return 0;
		}
	}
	return 1;
}

EXPORT int
vec2_any_one(void *ptr, size_t horz, size_t vert) {

	for (size_t i = 0; i < vert; i++) {
		if (vec1_all_ones(ptr, horz)) {
			return 1;
		}
		ptr = (char *)ptr + sizeof (__m256i) * horz;
	}
	return 0;
}

EXPORT size_t
vec2_cofactor_pos(void *ptr, size_t horz, size_t vert, size_t pos,
				  _short_t_ val) {
	size_t line_size = horz * sizeof (__m256i);
	void *curr_term = ptr;
	void *end = (char *)ptr + vert * line_size;
	
	val = element_negate(val);

	while (curr_term < end) {
		if (vec1_get_pos(curr_term, pos) == val) {
			while (end != curr_term) {
				end = (char *)end - line_size;
				if (vec1_get_pos(end, pos) != val) {
					break;
				}
			}
			if (curr_term == end) {
				goto VEC2_COFACTOR_EXIT;
			} else {
				vec1_copy(curr_term, end, horz);
				vec1_or_pos(curr_term, pos, val);
			}
		}
		curr_term = (char *)curr_term + line_size;
	}
VEC2_COFACTOR_EXIT:
	return (end - ptr) / line_size;
}

EXPORT size_t
vec2_and_pos(void *ptr, size_t horz, size_t vert, size_t pos, _short_t_ val) {
	size_t line_size = horz * sizeof (__m256i);
	void *curr_term = ptr;
	void *end = (char *)ptr + vert * line_size;
	
	while (curr_term < end) {
		if (vec1_and_pos(curr_term, pos, val) == 0) {
			while (end != curr_term) {
				end = (char *)end - line_size;
				if (vec1_and_pos(end, pos, val) != 0) {
					break;
				}
			}
			if (curr_term == end) {
				goto VEC2_AND_EXIT;
			} else {
				vec1_copy(curr_term, end, horz);
			}
		}
		curr_term = (char *)curr_term + line_size;
	}
VEC2_AND_EXIT:
	return (end - ptr) / line_size;
}


EXPORT void
vec2_sumdel(void *restrict ptr, size_t horz, size_t vert,
			int *restrict sum_beg, int *restrict del_beg) {
	uint16_t *vals = ptr;
	int *sum;
	int *del;
	__m256i positive_mask = _mm256_set_epi32(
		1 << 0xf, 1 << 0xd, 1 << 0xb, 1 << 0x9,
		1 << 0x7, 1 << 0x5, 1 << 0x3, 1 << 0x1);
	__m256i negative_mask = _mm256_set_epi32(
		1 << 0xe, 1 << 0xc, 1 << 0xa, 1 << 0x8,
		1 << 0x6, 1 << 0x4, 1 << 0x2, 1 << 0x0);
	__m256i zero = _mm256_setzero_si256();
	__m256i one = _mm256_set1_epi32(1);
	__m256i neg = _mm256_set1_epi32(-1);
	__m256i vals_repeat;
	__m256i positive_reg;
	__m256i negative_reg;
	__m256i positive_count;
	__m256i negative_count;
	__m256i sum_reg;
	__m256i del_reg;

	// 01 -> sum = sum - 1, del = del - 1 | 10 -> sum = sum - 1, del = del + 1
	for (size_t i = 0; i < vert; i++) {
		sum = sum_beg;
		del = del_beg;
		for (size_t j = 0; j < horz * (sizeof (__m256i) / sizeof (uint16_t));
			 j++) {
			vals_repeat = _mm256_set1_epi32(*vals);

			sum_reg = _mm256_load_si256((void *)sum);
			positive_reg = _mm256_and_si256(vals_repeat, positive_mask);		
			negative_reg = _mm256_and_si256(vals_repeat, negative_mask);		
			
			del_reg = _mm256_load_si256((void *)del);
			positive_count = _mm256_cmpeq_epi32(positive_reg, zero);
			negative_count = _mm256_cmpeq_epi32(negative_reg, zero);

			sum_reg = _mm256_add_epi32(sum_reg, positive_count);
			del_reg = _mm256_add_epi32(del_reg, positive_count);
			sum_reg = _mm256_add_epi32(sum_reg, negative_count);
			del_reg = _mm256_sub_epi32(del_reg, negative_count);

			_mm256_store_si256((void *)sum, sum_reg);
			_mm256_store_si256((void *)del, del_reg);
			
			sum += sizeof (__m256i) / sizeof (int);
			del += sizeof (__m256i) / sizeof (int);
			vals++;
		}
	}
	// sum = -sum, del = abs(del)
	sum = sum_beg;
	del = del_beg;
	for (size_t j = 0; j < horz * (sizeof (__m256i) / sizeof (uint16_t));
		 j++) {
			sum_reg = _mm256_load_si256((void *)sum);
			del_reg = _mm256_load_si256((void *)del);
			sum_reg = _mm256_xor_si256(sum_reg, neg);
			sum_reg = _mm256_add_epi32(sum_reg, one);
			del_reg = _mm256_abs_epi32(del_reg);

			_mm256_store_si256((void *)sum, sum_reg);
			_mm256_store_si256((void *)del, del_reg);

			sum += sizeof (__m256i) / sizeof (int);
			del += sizeof (__m256i) / sizeof (int);
	}
}

