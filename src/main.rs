#![feature(generic_arg_infer)]
extern crate image as im;
extern crate piston_window;
extern crate vecmath;
use std::{
    sync::mpsc::{channel, Receiver, Sender},
    thread::{self, spawn},
};

#[derive(Debug)]
struct PixelDrawInstruction {
    x: u32,
    y: u32,
    r: f32,
    g: f32,
    b: f32,
    width: u32,
    height: u32,
}

use im::{GenericImageView, ImageReader, Pixel};

mod matrix;
mod neuronal_network;

use jaime::{
    simd_arr::hybrid_simd::CriticalityCue,
    trainer::{
        asymptotic_gradient_descent_trainer::AsymptoticGradientDescentTrainer,
        default_param_translator, genetic_trainer::GeneticTrainer, DataPoint, Trainer,
    },
};

use piston_window::*;

use neuronal_network::neuronal_network;
use rand::seq::SliceRandom;
use tqdm::tqdm;
use types::Width;

fn main() {
    let (width, height) = (474, 316);

    let (tx, rx) = channel();
    let stack_size = 4 * 1024 * 1024 * 1024; // 4 MB

    let builder = thread::Builder::new()
        .name("worker".into())
        .stack_size(stack_size);

    let handle = builder.spawn(move || main_train(tx, width, height));

    let opengl = OpenGL::V3_2;
    let mut window: PistonWindow = WindowSettings::new("piston: paint", (width, height))
        .exit_on_esc(true)
        .graphics_api(opengl)
        .build()
        .unwrap();

    let mut canvas = im::ImageBuffer::new(width, height);
    let mut texture_context = TextureContext {
        factory: window.factory.clone(),
        encoder: window.factory.create_command_buffer().into(),
    };

    let mut settings = TextureSettings::new();
    settings.set_filter(Filter::Nearest);

    let mut texture: G2dTexture =
        Texture::from_image(&mut texture_context, &canvas, &settings).unwrap();

    let mut last_image_w = 0;
    let mut last_image_h = 0;
    while let Some(e) = window.next() {
        for img in rx.try_iter() {
            // println!("{img:?}");
            for pixel in img {
                last_image_h = pixel.height;
                last_image_w = pixel.width;

                canvas.put_pixel(
                    pixel.x,
                    pixel.y,
                    im::Rgba([
                        (pixel.r * 255.) as u8,
                        (pixel.g * 255.) as u8,
                        (pixel.b * 255.) as u8,
                        255,
                    ]),
                );
            }
        }

        if e.render_args().is_some() {
            texture.update(&mut texture_context, &canvas).unwrap();
            window.draw_2d(&e, |c, g, device| {
                // Update texture before rendering.
                texture_context.encoder.flush(device);

                clear([1.0; 4], g);
                image(
                    &texture,
                    c.transform.scale(
                        width as f64 / last_image_w as f64,
                        height as f64 / last_image_h as f64,
                    ),
                    g,
                );
            });
        }
    }
}

fn main_train(tx: Sender<Vec<PixelDrawInstruction>>, width: u32, height: u32) {
    let mut rng = rand::thread_rng();
    let structure = vec![2, 50, 50, 50, 3];

    let mut trainer: AsymptoticGradientDescentTrainer<5609, _, _, _, _, _, _, _> =
        AsymptoticGradientDescentTrainer::new_hybrid(
            CriticalityCue::<40>(),
            neuronal_network,
            neuronal_network,
            default_param_translator,
            structure,
        );

    // let mut trainer: GeneticTrainer<821, _, _, 100, 100, _, _, _> = GeneticTrainer::new(
    //     neuronal_network,
    //     default_param_translator,
    //     structure,
    //     0.1,
    //     10,
    // );

    let og_img = ImageReader::open("dog.jpg")
        .expect("error")
        .decode()
        .expect("error");

    for i in 1..101 {
        let downscaling = i as f32 / 100 as f32;
        let width = (width as f32 * downscaling) as u32;
        let height = (height as f32 * downscaling) as u32;
        println!("training for {}x{}", width, height);
        let img = og_img.resize(width, height, im::imageops::FilterType::CatmullRom);
        img.save(&format!("./train_levels/level{i}.jpg"))
            .expect("error saving level");
        let (width, height) = img.dimensions();
        let mut dataset = vec![];

        for y in 0..height {
            for x in 0..width {
                // Get the pixel at (x,y) and convert it to RGBA
                let pixel = img.get_pixel(x, y).to_rgba();
                let rgba = pixel.0;
                dataset.push(DataPoint {
                    input: [x as f32 / width as f32, y as f32 / height as f32],
                    output: [
                        (rgba[0]) as f32 / 255.,
                        (rgba[1]) as f32 / 255.,
                        (rgba[2]) as f32 / 255.,
                    ], //[],
                });
            }
        }

        // println!("dataset: {dataset:?}");

        trainer.found_local_minima = false;
        while !trainer.found_local_minima() {
            dataset.shuffle(&mut rng);

            trainer.train_step::<true, true, _, _>(
                &dataset,
                &dataset,
                dataset.len(),
                dataset.len(),
                0.1,
            );

            let mut img = vec![];

            for y in 0..height {
                for x in 0..width {
                    let prediction =
                        trainer.eval(&[x as f32 / width as f32, y as f32 / height as f32]);

                    // println!("{:?}", prediction);

                    img.push(PixelDrawInstruction {
                        r: prediction[0],
                        g: prediction[1],
                        b: prediction[2],
                        x,
                        y,
                        width,
                        height,
                    });
                }
            }

            tx.send(img).expect("send error");
        }
    }

    println!("training done!");
}
