from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.bfloat16, batch_size=16, max_length=430)

from jax.experimental.compilation_cache import compilation_cache as cc

cc.initialize_cache("./jax_cache")

text = pipeline("./streamyard_townhall.mp3", language="en", stride_length_s=0.0, return_timestamps=True) 

print(text)