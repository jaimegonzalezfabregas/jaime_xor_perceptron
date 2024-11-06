#![feature(generic_arg_infer)]
extern crate image as im;
extern crate piston_window;
extern crate vecmath;

mod matrix;
mod neuronal_network;

use jaime::{
    simd_arr::hybrid_simd::CriticalityCue,
    trainer::{
        asymptotic_gradient_descent_trainer::AsymptoticGradientDescentTrainer,
        default_param_translator, DataPoint, Trainer,
    },
};

use neuronal_network::neuronal_network;

fn main() {
    let dataset = vec![
        DataPoint {
            input: [0., 0.],
            output: [0.],
        },
        DataPoint {
            input: [1., 0.],
            output: [1.],
        },
        DataPoint {
            input: [0., 1.],
            output: [1.],
        },
        DataPoint {
            input: [1., 1.],
            output: [0.],
        },
    ];

    let mut trainer = AsymptoticGradientDescentTrainer::new_dense(
        neuronal_network,
        neuronal_network,
        default_param_translator,
        (),
    );

    while !trainer.found_local_minima() {
        trainer.train_step::<true, true, _, _>(&dataset, &dataset, 4, 4, 1.);
    }

    for dp in dataset {
        let prediction = trainer.eval(&dp.input);

        println!(
            "input {:?} goal {:?} prediction{:?}",
            dp.input, dp.output, prediction
        );
    }
}
