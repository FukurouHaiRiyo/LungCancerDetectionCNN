use std::fs;
use std::path::Path;
use rand::seq::SliceRandom;
use rand::thread_rng;
use ndarray::Array1;
use itertools::Itertools;

use std::path::PathBuf;
use std::collections::HashMap;

use image::imageops::FilterType;
use image::GenericImageView;
use csv::Reader;
use ndarray::Array3;
use ndarray::s;
use ndarray::stack;
use ndarray::ArrayD;
use ndarray::Axis;
use ndarray::Array;
use ndarray::Dimension;


fn train_test_split(df: Vec<String>, train_split: f32, rng: &mut impl rand::Rng) -> (Vec<String>, Vec<String>){
    let train_size = (df.len() as f32 * train_split) as usize;
    let shuffled_df = df.choose_multiple(rng, df.len()).cloned().collect_vec();
    let (train_df, test_df) = shuffled_df.split_at(train_size);

    (train_df.to_vec(), test_df.to_vec())
}

fn sorted_test_batch_size(length: usize) -> usize{
    let mut test_batch_sizes: Vec<usize> = (1..=length)
            .filter(|n| length % n == 0 && length / *n <= 80)
            .map(|n| length / n)
            .collect();
    
    test_batch_sizes.sort_unstable_by(|a, b| b.cmp(a));
    *test_batch_sizes.first().unwrap()
}

fn scalar(img: &mut [f64]){
    for pixel in img.iter_mut(){
        *pixel = *pixel / 127.5 - 1.0;
    }
}

struct ImageDataGenerator<F>
where F: Fn(usize) -> (Array3<f32>, String),
{
    loader: F,
    class_indices: HashMap<String, usize>,
    class_mode: String,
    color_mode: String,
    shuffle: bool,
    batch_size: usize,
}

impl<F> ImageDataGenerator<F>
where F: Fn(usize) -> (Array3<f32>, String),
{
    fn new(loader: F) -> Self{
        Self{
            loader,
            class_indices: HashMap::new(),
            class_mode: String::from(""),
            color_mode: String::from(""),
            shuffle: false,
            batch_size: 32,
        }
    }

    fn class_indices(mut self, class_indices: HashMap<String, usize>) -> Self {
        self.class_indices = class_indices;
        self
    }

    fn class_mode(mut self, class_mode: &str) -> Self {
        self.class_mode = String::from(class_mode);
        self
    }

    fn color_mode(mut self, color_mode: &str) -> Self {
        self.color_mode = String::from(color_mode);
        self
    }

    fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}

fn convert_image_to_array(image: &image::DynamicImage, color_mode: &str) -> Array3<f32>{
    let image = match color_mode{
        "rgb" => image.into_rgb16().into_raw(),
        "grayscale" => {
            
        }
        _ => panic!("Invalud color mode"),
    };

    let (width, height) = image.dimensions();
    let mut pixels = Vec::with_capacity((width * height) as usize);

    for(_, _, pixel) in image.pixels(){
        pixels.push(pixel[0] as f32/255.0);
    }

    let array = Array::from_shape_vec((height as usize, width as usize, 1), pixels).unwrap();
    array
}

fn flow_from_dataframe<'a, F>(
    df: &'a DataFrame, 
    x_col: &str,
    y_col: &str,
    target_size: (usize, usize),
    class_mode: &str,
    color_mode: &str,
    shuffle: bool,
    batch_size: usize,
) -> ImageDataGenerator<impl Fn(usize) -> (Array3<f32>, String) + 'a> 
{
    let filepaths = df[x_col].to_owned();
    let labels = df[y_col].to_owned();

    let filepaths_bufs: Vec<PathBuf> = filepaths.iter().map(|path| PathBuf::from(path)).collect();
    let class_indices: HashMap<String, usize> = classes.iter().enumerate().map(|(i, class)| (class.clone(), i)).collect();

    let gen = ImageDataGenerator::new(move |file_index| {
        let filepath = &filepaths_bufs[file_index];
        let label = &labels[file_index];
        let image = image::open(filepath).expect("Failed to open image file");

        let resized_image = image.resize_exact(target_size.0 as u32, target_size.1 as u32, FilterType::Lanczos3);
        let image_array = convert_image_to_array(&resized_image, color_mode);

        (image_array, label.clone())
    })
    .class_indices(class_indices)
    .class_mode(class_mode)
    .color_mode(color_mode)
    .shuffle(shuffle)
    .batch_size(batch_size);

    gen
}

fn main() {
    let path = Path::new("/home/andrei/Desktop/Licenta/lung_image_sets");
    let sample_size = 3000;

    let mut sample_list: Vec<Vec<String>> = Vec::new();

    if let Ok(classes) = fs::read_dir(path) {
        for class in classes {
            if let Ok(class_entry) = class {
                let class_path = class_entry.path();
                if class_path.is_dir() {
                    let file_paths: Vec<String> = fs::read_dir(class_path)
                        .unwrap()
                        .map(|entry| entry.unwrap().path().to_string_lossy().into_owned())
                        .collect();

                    let mut rng = thread_rng();
                    let sample_indices: Vec<usize> = (0..file_paths.len()).collect();
                    let shuffled_indices: Vec<usize> = sample_indices.choose_multiple(&mut rng, sample_size).cloned().collect();

                    let label_group: Vec<String> = shuffled_indices.iter()
                        .map(|&index| file_paths[index].clone())
                        .collect();
                    
                    sample_list.push(label_group);
                }
            }
        }
    }

    let df: Vec<String> = sample_list.into_iter().flatten().collect();
    println!("{}", df.len());

    let train_split: f32 = 0.8;
    let test_split: f32 = 0.1;
    let valid_split: f32 = test_split / (1.0 - train_split);

    let mut rng = thread_rng();

    let (train_df, dummy_df) = train_test_split(df, train_split, &mut rng);
    let (test_df, valid_df) = train_test_split(dummy_df, valid_split, &mut rng);

    println!("Train length: {}\nTest length: {}\nValidation length: {}", train_df.len(), test_df.len(), valid_df.len());

    let (height, width, channels, batch_size) = (224, 224, 3, 32);
    let img_shape = (height, width, channels);
    let img_size = (height, width);
    let length = test_df.len();
    let test_batch_size = sorted_test_batch_size(length);
    let test_steps = length / test_batch_size;

    println!("Test batch size: {}; Test steps: {}", test_batch_size, test_steps);
}
