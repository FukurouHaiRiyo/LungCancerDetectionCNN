use image::{DynamicImage, GenericImageView};
use std::path::Path;
use tch::nn::{conv2d, Func, Module, ModuleT, OptimizerConfig};
use tch::{Device, Kind, Tensor, Vision};

struct SimpleCNN {
    conv1: conv2d::Conv2D,
    conv2: conv2d::Conv2D,
    conv3: conv2d::Conv2D,
    fc: Func,
}

impl SimpleCNN {
    fn new(vs: &tch::nn::Path) -> Self {
        let conv1 = conv2d(vs / "conv1", 3, 32, 3, Default::default());
        let conv2 = conv2d(vs / "conv2", 32, 64, 3, Default::default());
        let conv3 = conv2d(vs / "conv3", 64, 128, 3, Default::default());
        let fc = Func::new(vs / "fc", &[128, 3], Kind::Float, Default::default());

        SimpleCNN { conv1, conv2, conv3, fc }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        let x = xs.view([-1, 3, 224, 224]);
        let x = self.conv1.forward(&x.relu());
        let x = self.conv2.forward(&x.relu());
        let x = self.conv3.forward(&x.relu());
        let x = x.flatten(1);
        self.fc.forward(&x)
    }
}

fn load_image(path: &str) -> DynamicImage {
    image::open(&Path::new(path)).expect("Failed to open image")
}

fn load_dataset(path: &str) -> Vec<DynamicImage> {
    let mut images = Vec::new();
    let entries = std::fs::read_dir(path).expect("Failed to read directory");

    for entry in entries {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();

        if let Some(extension) = path.extension() {
            if extension == "jpg" || extension == "png" {
                let image = load_image(path.to_str().unwrap());
                images.push(image);
            }
        }
    }

    images
}

fn preprocess_images(images: &[DynamicImage]) -> Vec<Tensor> {
    let mut processed_images = Vec::new();

    for image in images {
        let resized_image = image.resize_exact(224, 224, image::imageops::FilterType::Gaussian);
        let tensor = Tensor::of_slice(resized_image.to_bytes().as_slice())
            .reshape(&[1, 224, 224, 3])
            .to_kind(Kind::Float);
        processed_images.push(tensor);
    }

    processed_images
}

fn train_model(model: &SimpleCNN, train_data: &[Tensor], train_labels: &[i64]) -> Result<(), Box<dyn std::error::Error>> {
    let vs = tch::nn::VarStore::new(Device::Cpu);
    let net = SimpleCNN::new(&vs.root());
    let mut opt = tch::nn::Adam::default().build(&vs, 1e-3)?;

    let num_epochs = 10;
    let log_interval = 100;

    for epoch in 0..num_epochs {
        for (data, label) in train_data.iter().zip(train_labels.iter()) {
            let output = net.forward(data);
            let loss = output.cross_entropy_for_logits(&Tensor::of_slice(&[*label]));

            opt.backward_step(&loss);
        }

        if epoch % log_interval == 0 {
            println!("Epoch: {:4} Loss: {}", epoch, loss.double_value(&[]));
        }
    }

    Ok(())
}

fn load_train_labels(path: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let image_files = std::fs::read_dir(path)?;
    let mut train_labels = Vec::new();

    for image_file in image_files {
        let file_name = image_file?.file_name();
        let label = file_name.to_string_lossy().to_string();
        train_labels.push(label);
    }

    Ok(train_labels)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let train_data_path = load_dataset("/home/andrei/Desktop/Licenta/lung_image_sets");
    let train_labels = load_train_labels(train_data_path)?;
    let preprocessed_train_data = preprocess_images(&train_data);

    train_model(&net, &preprocessed_train_data, &train_labels)?;

    Ok(())
}

