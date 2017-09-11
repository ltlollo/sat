#ifndef LOGIC_PRIMITIVES_H
#define LOGIC_PRIMITIVES_H

#include <stdint.h>
#include <stddef.h>

#ifndef EXPORT
#ifdef __cplusplus
#define EXPORT extern "C"
#else
#define EXPORT extern
#endif
#endif

#ifdef __cplusplus
#define restrict
#endif

typedef unsigned short _short_t_; 

EXPORT void vec1_fill_ones(void *, size_t);
EXPORT void vec1_dill_var_pos(void *, size_t, size_t, _short_t_);
EXPORT void vec1_copy(void *restrict, void *restrict, size_t);
EXPORT void vec1_var_pos(void *, size_t, size_t,  _short_t_);
EXPORT _short_t_ vec1_all_ones(void *, size_t);
EXPORT int vec2_any_one(void *, size_t, size_t);
EXPORT size_t vec2_cofactor_pos(void *, size_t, size_t, size_t, _short_t_);
EXPORT size_t vec2_and_pos(void *, size_t, size_t, size_t, _short_t_);
EXPORT void vec2_sumdel(void *restrict , size_t, size_t, int *restrict,
						int *restrict);

#endif // LOGIC_PRIMITIVES_H
