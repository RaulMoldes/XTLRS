// aligned_vec.rs

// Importar las librerías necesarias
use std::alloc::{self, Layout};
use std::vec::Vec;
use std::{mem, ptr};

macro_rules! create_aligned_vec {
    ($name:ident, $align:expr) => {
        pub struct $name<T: Copy> {
            pub data: *mut T,
            pub len: usize,
            capacity: usize,
        }

        impl<T: Copy> $name<T> {
            pub fn new_from_value(size: usize, value: T) -> Self {
                let layout = Layout::from_size_align(size * mem::size_of::<T>(), $align).unwrap();
                let data = unsafe { alloc::alloc(layout) as *mut T };

                if data.is_null() {
                    panic!("Error al asignar memoria alineada");
                }

                // Rellenar con el valor proporcionado
                for i in 0..size {
                    unsafe {
                        ptr::write(data.add(i), value);
                    }
                }

                $name {
                    data,
                    len: size,
                    capacity: size,
                }
            }

            pub fn as_ptr(&self) -> *const T {
                self.data as *const T
            }

            pub fn as_mut_ptr(&mut self) -> *mut T {
                self.data
            }

            pub fn len(&self) -> usize {
                self.len
            }
        }

        impl<T: Copy> Drop for $name<T> {
            fn drop(&mut self) {
                unsafe {
                    let layout =
                        Layout::from_size_align(self.capacity * mem::size_of::<T>(), $align)
                            .unwrap();
                    alloc::dealloc(self.data as *mut u8, layout);
                }
            }
        }
    };
}

// Crear tipos con alineación de 16 y 32 bytes
create_aligned_vec!(AlignedVec16, 16);
create_aligned_vec!(AlignedVec32, 32);
