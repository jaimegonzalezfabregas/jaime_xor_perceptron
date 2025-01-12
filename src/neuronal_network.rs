use std::fmt::Debug;
use std::ops::{Add, Div, Mul};

use jaime::dual::extended_arithmetic::ExtendedArithmetic;

use crate::matrix::Matrix;

fn sigmoid<N: ExtendedArithmetic + Debug>(x: &mut N) {
    x.sigmoid_on_mut();
}

pub fn neuronal_network<
    N: Clone
        + Debug
        + From<f32>
        + PartialOrd<f32>
        + PartialOrd<N>
        + Add<N, Output = N>
        + Mul<N, Output = N>
        + Mul<f32, Output = N>
        + Div<N, Output = N>
        + ExtendedArithmetic,
>(
    params: &[N; 13],
    input: &[f32; 2],
    _: &(),
) -> [N; 1] {

    // hardcoded 2 - 3 - 1 neuronal network. for a more flexible aproach using the extra data to define the network layers see https://github.com/jaimegonzalezfabregas/jaime_mnist_perceptron/blob/main/src/neuronal_network.rs

    let first_layer_weights = Matrix::deserialize(
        3, // 2 + a bias
        3,
        &params[0..9],
    );

    let second_layer_weights = Matrix::deserialize(
        4, // 2 + a bias
        1,
        &params[9..13],
    );

    let input = Matrix::deserialize(1, 2, &input.map(N::from));

    let mut hidden_layer_values = first_layer_weights * input.add_bias();
    hidden_layer_values.delinearize(sigmoid);

    let mut output = second_layer_weights * hidden_layer_values.add_bias();
    output.delinearize(sigmoid);

    return [output.serialize()[0].clone()];
}
