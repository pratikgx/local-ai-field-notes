# Glossary

The terms that confused me when I started, in plain words.

## The basics

**Model.** A large file of numbers learned during training. Running the model means doing math on those numbers to predict the next piece of text.

**Weights.** The numbers inside the model. "Downloading a model" means downloading its weights.

**Parameters.** How many numbers the model has. An 8B model has about 8 billion. More parameters usually means smarter and slower, and it always means more memory.

**Token.** The unit a model reads and writes. A token is roughly three quarters of an English word.

**Tokens per second.** How fast a model writes. Around 8 or more feels usable for chat. 30 or more feels quick.

**Context window.** How much text the model can consider at once, measured in tokens. Longer context uses more memory.

**Inference.** Running a model to get an answer. Training is how a model gets made, and inference is using it.

## Open source and open weight

**Open-weight model.** A model whose weights you can download and run yourself. The training data and training code may stay private.

**Open-source AI.** A stricter idea. The Open Source Initiative's definition asks for the weights, the code used to train and run the model, and enough detail about the training data that someone skilled could rebuild a similar system. Most models people call open source are really open weight.

## Running models on your own machine

**Local inference.** Running the model on hardware you own instead of calling a company's servers.

**Runtime or inference engine.** The software that loads the weights and generates text. llama.cpp, MLX, Ollama and vLLM are runtimes.

**llama.cpp.** The open-source engine much of local AI is built on. Georgi Gerganov started it in 2023 to run models on ordinary hardware, and it now runs on Macs, Windows, Linux and even phones. It created the GGUF format. It's a command-line tool for developers, and many friendlier apps use it underneath.

**MLX.** Apple's framework for running and training models on Apple silicon. It only works on Macs, and on a Mac it's usually the fastest option. Models for it come in their own MLX format, often from the mlx-community group on Hugging Face.

**mlx-lm.** The command-line tool for running language models with MLX. `mlx_lm.chat` opens a chat in the terminal, and `mlx_lm.generate` answers a single prompt and reports tokens per second and peak memory. It's what scripts/bench.py uses.

**Ollama.** A free app that makes running a model a one-command job: type `ollama run` and a model name, and it downloads and starts the model for you. It has its own model library, a simple chat window and a local server other apps can connect to. Underneath it uses a llama.cpp-style engine and GGUF files, so it can't use MLX downloads.

**LM Studio.** A free desktop app with a chat window, a built-in model browser that searches Hugging Face, and a local server. It runs both GGUF models through llama.cpp and MLX models on Macs, which makes it a good starting point if you don't want to use the terminal.

**Hugging Face.** The website where most open models, quantized versions and datasets are published.

**Tailscale.** An app that puts your own devices on a private network, so you can reach your Mac from your laptop or phone wherever you are. It's a kind of VPN, but by default it only carries traffic between your devices. Your normal internet traffic only goes through it if you turn on an exit node.

## Memory

**VRAM.** Memory on a graphics card. On a PC, the model has to fit in VRAM to run fast.

**Unified memory.** On Apple silicon, the CPU and GPU share one pool of memory. That's why a Mac with lots of memory can run models too big for most single graphics cards.

**Rule of thumb.** At 4-bit quantization, a model needs a bit more than half a gigabyte of memory per billion parameters, plus room for context. An 8B model fits comfortably in 8 GB. A 27B model wants about 16 GB or more.

## Making models smaller

**Quantization.** Storing each weight with fewer bits so the model takes less memory. Full precision is usually 16 bits per weight. 4-bit quantization cuts the size to roughly a quarter, and quality drops less than you'd expect.

**GGUF.** The file format llama.cpp uses for quantized models. Names like Q4_K_M describe the quantization level, where Q4 means about 4 bits per weight.

**MoE (mixture of experts).** A model built from many smaller "expert" parts, where only some run for each token. A 35B MoE with 3B active parameters needs memory for all 35B but runs about as fast as a 3B model.

## Changing what a model does

**Fine-tuning.** Training an existing model a little more on your own examples so it gets better at a specific job.

**LoRA and QLoRA.** Cheap ways to fine-tune. LoRA trains a small add-on instead of the whole model. QLoRA does the same on a 4-bit quantized model, which made fine-tuning possible on a single consumer GPU.

**Distillation.** Training a small model to imitate a bigger one.

**Abliterated.** A model with the trained-in refusal behavior removed from its weights, so it answers requests it would normally decline.
