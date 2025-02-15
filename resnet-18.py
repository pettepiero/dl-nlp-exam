import jax
import jax.numpy as jnp
import haiku as hk
import optax
import tensorflow_datasets as tfds
import numpy as np


# Load CIFAR-10 dataset
def load_cifar10():
    ds_builder = tfds.builder("cifar10")
    ds_builder.download_and_prepare()

    def preprocess(example):
        image = example["image"].astype(np.float32) / 255.0  # Normalize to [0,1]
        label = example["label"]
        return image, label

    def prepare_tf_dataset(split, batch_size):
        ds = tfds.load("cifar10", split=split, as_supervised=True)
        ds = ds.map(preprocess).shuffle(10_000).batch(batch_size).prefetch(1)
        return ds

    train_ds = prepare_tf_dataset("train", batch_size=128)
    test_ds = prepare_tf_dataset("test", batch_size=128)

    return train_ds, test_ds


# ResNet-18 Building Blocks
class ResNetBlock(hk.Module):
    def __init__(self, channels, stride=1, use_projection=False):
        super().__init__()
        self.use_projection = use_projection
        self.bn1 = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)
        self.bn2 = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)

        self.conv1 = hk.Conv2D(
            channels, kernel_shape=3, stride=stride, padding="SAME", with_bias=False
        )
        self.conv2 = hk.Conv2D(
            channels, kernel_shape=3, stride=1, padding="SAME", with_bias=False
        )

        if use_projection:
            self.projection = hk.Conv2D(
                channels, kernel_shape=1, stride=stride, padding="SAME", with_bias=False
            )

    def __call__(self, x, is_training):
        shortcut = x
        if self.use_projection:
            shortcut = self.projection(shortcut)

        x = self.conv1(x)
        x = self.bn1(x, is_training)
        x = jax.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, is_training)

        x += shortcut
        return jax.nn.relu(x)


# ResNet-18 Model
class ResNet18(hk.Module):
    def __init__(self, num_classes=10):
        super().__init__()

    def __call__(self, x, is_training):
        x = hk.Conv2D(64, kernel_shape=3, stride=1, padding="SAME", with_bias=False)(x)
        x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(
            x, is_training
        )
        x = jax.nn.relu(x)

        def make_layer(channels, num_blocks, stride):
            layers = []
            layers.append(ResNetBlock(channels, stride=stride, use_projection=True))
            for _ in range(1, num_blocks):
                layers.append(ResNetBlock(channels, stride=1))
            return layers

        layers = (
            make_layer(64, 2, stride=1)
            + make_layer(128, 2, stride=2)
            + make_layer(256, 2, stride=2)
            + make_layer(512, 2, stride=2)
        )

        for layer in layers:
            x = layer(x, is_training)

        x = jnp.mean(x, axis=(1, 2))  # Global Average Pooling
        x = hk.Linear(10)(x)  # Fully connected layer
        return x


# Forward function
def forward_fn(x, is_training):
    model = ResNet18()
    return model(x, is_training)


# Transform the model with Haiku
resnet_model = hk.transform_with_state(forward_fn)


# Loss function
def loss_fn(params, state, images, labels):
    logits, new_state = resnet_model.apply(
        params, state, None, images, is_training=True
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    return loss, new_state


# Training step
@jax.jit
def train_step(params, state, opt_state, images, labels, opt_update):
    (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, state, images, labels
    )
    updates, new_opt_state = opt_update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_state, new_opt_state, loss


# Accuracy function
@jax.jit
def compute_accuracy(params, state, images, labels):
    logits, _ = resnet_model.apply(params, state, None, images, is_training=False)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)


# Training loop
def train():
    train_ds, test_ds = load_cifar10()

    # Initialize model and optimizer
    rng = jax.random.PRNGKey(42)
    sample_input = jnp.zeros((1, 32, 32, 3))
    params, state = resnet_model.init(rng, sample_input, is_training=True)

    opt_init, opt_update = optax.adam(1e-3)
    opt_state = opt_init(params)

    # Training loop
    for epoch in range(20):  # 20 epochs for better accuracy
        total_loss = 0.0
        for images, labels in train_ds:
            images, labels = jnp.array(images), jnp.array(labels)
            params, state, opt_state, loss = train_step(
                params, state, opt_state, images, labels, opt_update
            )
            total_loss += loss

        # Compute accuracy on test set
        test_acc = 0.0
        num_batches = 0
        for images, labels in test_ds:
            images, labels = jnp.array(images), jnp.array(labels)
            test_acc += compute_accuracy(params, state, images, labels)
            num_batches += 1
        test_acc /= num_batches

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Test Accuracy: {test_acc:.4f}")


train()
