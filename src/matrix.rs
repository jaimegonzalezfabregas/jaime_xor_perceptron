use std::fmt::Debug;
use std::ops::Mul;

use jaime::dual::extended_arithmetic::ExtendedArithmetic;
use rand::Rng;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T>(Vec<Vec<T>>);

impl<N: From<f32> + Clone + Debug> Matrix<N> {
    pub fn delinearize<F: Fn(&mut N)>(&mut self, delinearizer: F) {
        for y in 0..self.0.len() {
            for x in 0..self.0[y].len() {
                delinearizer(&mut self.0[y][x]);
            }
        }
    }

    pub fn new<const W: usize, const H: usize>(data: [[N; W]; H]) -> Self {
        let data = data
            .iter()
            .map(|row| row.to_vec()) // Convert each row from an array to a Vec
            .collect(); // Collect into a Vec<Vec<N>>

        Matrix(data)
    }

    pub fn cero(width: usize, height: usize) -> Self {
        let data = (0..height)
            .map(|_| (0..width).map(|_| N::from(0.)).collect())
            .collect();
        Matrix(data)
    }

    pub fn unit(size: usize) -> Self {
        let mut ret = Self::cero(size, size);
        for i in 0..size {
            ret.0[i][i] = N::from(1.0);
        }
        ret
    }

    pub fn deserialize(width: usize, height: usize, flat_data: &[N]) -> Self {
        assert_eq!(flat_data.len(), width * height);

        Matrix(
            (0..height)
                .map(|y| {
                    (0..width)
                        .map(|x| flat_data[y * width + x].clone())
                        .collect()
                })
                .collect(),
        )
    }

    // Pretty print the matrix
    pub fn pretty_print(&self) {
        for row in &self.0 {
            let row_str: Vec<String> = row.iter().map(|val| format!("{:?}", val)).collect();
            println!("[{}]", row_str.join(", "));
        }
    }

    pub fn add_bias(mut self) -> Matrix<N> {
        self.0.push(vec![N::from(1.)]);

        self
    }

    pub fn add_reminder<const X: usize>(mut self, reminder: &[N; X]) -> Matrix<N> {
        for n in reminder {
            self.0.push(vec![n.clone()]);
        }

        self
    }

    pub fn serialize(self) -> Vec<N> {
        self.0.into_iter().flat_map(|x| x).collect()
    }

    pub fn rows(&self) -> usize {
        self.0.len()
    }

    pub fn cols(&self) -> usize {
        self.0[0].len()
    }
}

impl Matrix<f32> {
    pub fn rand<R: Rng>(width: usize, height: usize, generator: &mut R) -> Self {
        let data: Vec<Vec<f32>> = (0..height)
            .map(|_| (0..width).map(|_| generator.gen()).collect())
            .collect();
        Matrix(data)
    }
}

impl<T> Mul for Matrix<T>
where
    T: std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + From<f32>
        + Clone
        + ExtendedArithmetic
        + Debug,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: Matrix<T>) -> Matrix<T> {
        let m = self.rows();
        let n = self.cols();
        let p = rhs.cols();

        // Ensure the matrices can be multiplied
        assert_eq!(
            n,
            rhs.rows(),
            "Incompatible matrix sizes for multiplication"
        );

        // Create a new matrix to hold the result
        let mut ret: Matrix<T> = Matrix::cero(p, m);

        // Perform matrix multiplication
        for i in 0..m {
            for j in 0..p {
                for k in 0..n {
                    ret.0[i][j].accumulate(&(self.0[i][k].clone() * rhs.0[k][j].clone()))
                }
            }
        }

        ret
    }
}

#[cfg(test)]
mod matrix_tests {

    use super::Matrix;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaChaRng;

    #[test]
    fn stress_test_multiplication() {
        let mut rand = ChaChaRng::seed_from_u64(0);

        for _ in 0..100 {
            let unit3 = Matrix::unit(3);
            let unit4 = Matrix::unit(3);
            let a: Matrix<f32> = Matrix::rand(3, 3, &mut rand);
            let b: Matrix<f32> = Matrix::rand(3, 3, &mut rand);

            assert_eq!(a, a.clone() * unit3.clone());
            assert_eq!(a, unit4 * a.clone());

            let ab = a.clone() * b.clone();

            assert_eq!(ab, a.clone() * (unit3.clone() * b.clone()));
            assert_eq!(ab, (a.clone() * unit3) * b.clone());
        }
    }

    #[test]
    fn test_multiplication_a() {
        let a = Matrix::new([[1., 2., 3.], [1., 0., 3.], [0., 2., 3.]]);
        let b = Matrix::new([[3., 2., 3.], [1., 3., 2.], [7., 6., 3.]]);

        let ab = Matrix::new([[26., 26., 16.], [24., 20., 12.], [23., 24., 13.]]);

        assert_eq!(a * b, ab);
    }

    #[test]
    fn test_multiplication_b() {
        let a = Matrix::new([[1., 2.], [3., 4.], [5., 6.]]);
        let b = Matrix::new([[7., 8., 9., 10.], [11., 12., 13., 14.]]);

        let ab = Matrix::new([
            [29., 32., 35., 38.],
            [65., 72., 79., 86.],
            [101., 112., 123., 134.],
        ]);

        assert_eq!(a * b, ab);
    }

    #[test]
    fn test_multiplication_c() {
        let a = Matrix::new([[1., 2., 3.], [3., 4., 0.]]);
        let b = Matrix::new([[7., 8.], [7., 9.], [9., 8.]]);

        let ab = Matrix::new([[48., 50.], [49., 60.]]);

        assert_eq!(a * b, ab);
    }
}
