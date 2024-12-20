mod alignedvec; // Importar el módulo alignedvec

use alignedvec::{AlignedVec16, AlignedVec32};
use std::arch::x86_64::{
    __m128i, __m256i, _mm256_add_epi32, _mm256_load_si256, _mm256_store_si256, _mm_add_epi32,
    _mm_load_si128, _mm_store_si128,
};
use std::mem;
use std::time::Instant;

fn sum_with_simd_avx<T>(a: &AlignedVec32<T>, b: &AlignedVec32<T>, result: &mut AlignedVec32<T>)
where
    T: Copy + std::ops::Add<Output = T>,
{
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());

    unsafe {
        let a_ptr = a.as_ptr() as usize;
        let b_ptr = b.as_ptr() as usize;
        let result_ptr = result.as_mut_ptr() as usize;

        assert_eq!(a_ptr % 32, 0, "El puntero `a` no está alineado a 32 bytes");
        assert_eq!(b_ptr % 32, 0, "El puntero `b` no está alineado a 32 bytes");
        assert_eq!(
            result_ptr % 32,
            0,
            "El puntero `result` no está alineado a 32 bytes"
        );
    }

    let chunks = a.len() / 8;

    unsafe {
        for i in 0..chunks {
            // Aseguramos que la alineación sea de 32 bytes
            let a_simd: __m256i = _mm256_load_si256(a.as_ptr().add(i * 8) as *const __m256i);
            let b_simd: __m256i = _mm256_load_si256(b.as_ptr().add(i * 8) as *const __m256i);

            // Sumamos los elementos
            let sum_simd: __m256i = _mm256_add_epi32(a_simd, b_simd);

            // Almacenamos el resultado
            _mm256_store_si256(result.as_mut_ptr().add(i * 8) as *mut __m256i, sum_simd);
        }

        // Manejo de los elementos restantes si el tamaño no es múltiplo de 8
        for i in (chunks * 8)..a.len() {
            *result.data.add(i) = *a.data.add(i) + *b.data.add(i);
        }
    }
}

fn sum_with_simd_aligned<T>(a: &AlignedVec16<T>, b: &AlignedVec16<T>, result: &mut AlignedVec16<T>)
where
    T: Copy + std::ops::Add<Output = T>,
{
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());

    let chunks = a.len() / 4;

    unsafe {
        for i in 0..chunks {
            let a_simd: __m128i = _mm_load_si128(a.as_ptr().add(i * 4) as *const __m128i);
            let b_simd: __m128i = _mm_load_si128(b.as_ptr().add(i * 4) as *const __m128i);
            let sum_simd: __m128i = _mm_add_epi32(a_simd, b_simd);
            _mm_store_si128(result.as_mut_ptr().add(i * 4) as *mut __m128i, sum_simd);
        }

        for i in (chunks * 4)..a.len() {
            *result.data.add(i) = *a.data.add(i) + *b.data.add(i);
        }
    }
}

fn main() {
    let size: usize = 1_000_000_000; // Un millón de elementos
    let a16 = AlignedVec16::<i32>::new_from_value(size, 1);
    let b16 = AlignedVec16::<i32>::new_from_value(size, 2);

    // Medir tiempo para SIMD normal (128 bits)
    let start_simd = Instant::now();
    let mut result_simd16 = AlignedVec16::<i32>::new_from_value(size, 0);
    sum_with_simd_aligned::<i32>(&a16, &b16, &mut result_simd16);
    let duration_simd = start_simd.elapsed();
    println!("Tiempo con SIMD 128 bits: {:?}", duration_simd);

    // Verificar que los resultados son correctos
    // Verificar que los resultados son correctos para 128 bits
    unsafe {
        let mut correct = true;
        for i in 0..size {
            if *result_simd16.data.wrapping_add(i) != 3 {
                correct = false;
                println!(
                    "Error en la posición {}: esperado 3, obtenido {}",
                    i,
                    *result_simd16.data.wrapping_add(i)
                );
                break;
            }
        }

        if correct {
            println!("¡Resultados correctos para el SIMD normal de 128 bits!");
        } else {
            println!("Se encontraron errores en los resultados para SIMD de 128 bits.");
        }
    }
    // Medir tiempo para SIMD AVX (256 bits)
    let start_simd = Instant::now();

    let a32 = AlignedVec32::<i32>::new_from_value(size, 1);
    let b32 = AlignedVec32::<i32>::new_from_value(size, 2);
    // Verificación de alineación en tiempo de ejecución

    let mut result_simd32 = AlignedVec32::<i32>::new_from_value(size, 0);
    sum_with_simd_avx::<i32>(&a32, &b32, &mut result_simd32);
    let duration_simd = start_simd.elapsed();
    println!("Tiempo con SIMD AVX 256 bits: {:?}", duration_simd);

    // Verificar que los resultados son correctos
    unsafe {
        let mut correct = true;
        for i in 0..size {
            if *result_simd32.data.wrapping_add(i) != 3 {
                correct = false;
                println!(
                    "Error en la posición {}: esperado 3, obtenido {}",
                    i,
                    *result_simd32.data.wrapping_add(i)
                );
                break;
            }
        }

        if correct {
            println!("¡Resultados correctos para el SIMD AVX de 256  bits!");
        } else {
            println!("Se encontraron errores en los resultados para SIMD AVX de 256  bits.");
        }
    }
}
