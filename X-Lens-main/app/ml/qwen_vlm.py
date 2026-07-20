from __future__ import annotations

import gc
from time import perf_counter
from typing import Any

from PIL import Image

from app.ml.base_vlm import MemoryStats, VLMResult


class QwenVLM:
    """
    Qwen2.5-VL adapter supporting text-only and image-based requests.

    The model is loaded using 4-bit NF4 quantization so it can fit
    more completely on a 6 GB NVIDIA GPU without CPU offloading.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.model = None
        self.processor = None
        self._torch = None

    def load(self) -> None:
        """
        Load Qwen2.5-VL using bitsandbytes 4-bit quantization.
        """

        try:
            import torch
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                Qwen2_5_VLForConditionalGeneration,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install the required dependencies with:\n"
                "python -m pip install --upgrade "
                "transformers accelerate bitsandbytes"
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. An NVIDIA CUDA GPU is required."
            )

        self._torch = torch

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        try:
            self.model = (
                Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    quantization_config=quantization_config,
                    device_map={"": 0},
                    attn_implementation="sdpa",
                    low_cpu_mem_usage=True,
                )
            )
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()

            raise RuntimeError(
                "The quantized Qwen model could not fit in GPU memory. "
                "Close GPU-heavy applications and restart X-Lens."
            ) from exc

        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )

        print("Qwen loaded in 4-bit mode.")
        print(
            "Qwen device map:",
            getattr(self.model, "hf_device_map", None),
        )
        print("Qwen memory:", self.get_memory_stats())

    def warmup(self) -> None:
        """
        Warm up text and image generation.
        """

        if self.model is None or self.processor is None:
            return

        try:
            self.generate_text(
                prompt="Reply only with: ready",
                max_new_tokens=3,
            )

            self.generate_vision(
                image=Image.new(
                    mode="RGB",
                    size=(64, 64),
                    color="white",
                ),
                prompt="Reply only with: ready",
                max_new_tokens=3,
            )

            self._torch.cuda.synchronize()

        except Exception as exc:
            print(f"Qwen warmup warning: {exc}")

    def generate(
        self,
        image: Any = None,
        prompt: str = "",
        max_new_tokens: int = 64,
    ) -> VLMResult:
        """
        Automatically choose text or vision generation.
        """

        if image is None:
            return self.generate_text(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )

        return self.generate_vision(
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 32,
    ) -> VLMResult:
        """
        Generate a text-only response.
        """

        self._ensure_loaded()

        clean_prompt = prompt.strip()

        if not clean_prompt:
            raise ValueError("The text prompt cannot be empty.")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": clean_prompt,
                    }
                ],
            }
        ]

        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        processor_inputs = self.processor(
            text=[chat_text],
            padding=False,
            return_tensors="pt",
        )

        inputs = self._move_inputs_to_gpu(processor_inputs)

        self._torch.cuda.synchronize()
        start = perf_counter()

        with self._torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                num_beams=1,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        self._torch.cuda.synchronize()
        elapsed_ms = (perf_counter() - start) * 1000

        input_ids = inputs["input_ids"]
        input_length = input_ids.shape[1]

        generated_tokens = generated[:, input_length:]

        answer = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return VLMResult(
            answer,
            int(input_ids.numel()),
            int(generated_tokens.numel()),
            elapsed_ms,
        )

    def generate_vision(
        self,
        image: Any,
        prompt: str,
        max_new_tokens: int = 40,
    ) -> VLMResult:
        """
        Generate a response using an image and text prompt.
        """

        self._ensure_loaded()

        if image is None:
            raise ValueError(
                "An image is required for vision generation."
            )

        clean_prompt = prompt.strip()

        if not clean_prompt:
            clean_prompt = (
                "Describe the important visible information."
            )

        prepared_image = self._prepare_image(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": prepared_image,
                    },
                    {
                        "type": "text",
                        "text": clean_prompt,
                    },
                ],
            }
        ]

        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        processor_inputs = self.processor(
            text=[chat_text],
            images=[prepared_image],
            padding=False,
            return_tensors="pt",
        )

        inputs = self._move_inputs_to_gpu(processor_inputs)

        self._torch.cuda.synchronize()
        start = perf_counter()

        with self._torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                num_beams=1,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        self._torch.cuda.synchronize()
        elapsed_ms = (perf_counter() - start) * 1000

        input_ids = inputs["input_ids"]
        input_length = input_ids.shape[1]

        generated_tokens = generated[:, input_length:]

        answer = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return VLMResult(
            answer,
            int(input_ids.numel()),
            int(generated_tokens.numel()),
            elapsed_ms,
        )

    def _prepare_image(
        self,
        image: Any,
    ) -> Image.Image:
        """
        Convert the image to RGB and limit oversized images.
        """

        if not isinstance(image, Image.Image):
            raise TypeError(
                "The image must be a PIL.Image.Image instance."
            )

        prepared = image.convert("RGB")

        max_side = 672

        if max(prepared.size) > max_side:
            prepared.thumbnail(
                (max_side, max_side),
                Image.Resampling.LANCZOS,
            )

        return prepared

    def _move_inputs_to_gpu(
        self,
        inputs: Any,
    ) -> dict[str, Any]:
        """
        Move processor tensors to CUDA.

        A normal dictionary is returned, so its values must later be
        accessed with inputs["input_ids"], not inputs.input_ids.
        """

        moved: dict[str, Any] = {}

        for name, value in inputs.items():
            if not hasattr(value, "to"):
                moved[name] = value
                continue

            if name in {
                "pixel_values",
                "pixel_values_videos",
            }:
                moved[name] = value.to(
                    device="cuda:0",
                    dtype=self._torch.float16,
                    non_blocking=True,
                )
            else:
                moved[name] = value.to(
                    device="cuda:0",
                    non_blocking=True,
                )

        return moved

    def get_memory_stats(self) -> MemoryStats:
        """
        Return CUDA memory use in megabytes.
        """

        if (
            self._torch is None
            or not self._torch.cuda.is_available()
        ):
            return MemoryStats()

        cuda = self._torch.cuda
        megabyte = 1024**2

        return MemoryStats(
            cuda.memory_allocated() / megabyte,
            cuda.memory_reserved() / megabyte,
            cuda.max_memory_allocated() / megabyte,
        )

    def unload(self) -> None:
        """
        Release model and CUDA resources.
        """

        self.model = None
        self.processor = None

        gc.collect()

        if (
            self._torch is not None
            and self._torch.cuda.is_available()
        ):
            self._torch.cuda.empty_cache()
            self._torch.cuda.ipc_collect()

    def _ensure_loaded(self) -> None:
        """
        Ensure the model and processor are loaded.
        """

        if self.model is None or self.processor is None:
            raise RuntimeError("Qwen is not loaded.")