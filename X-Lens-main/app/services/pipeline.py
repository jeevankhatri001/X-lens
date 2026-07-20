from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.services.request_router import RequestMode, RequestRouter
from app.utils.timing import timer


class XLensPipeline:
    """
    Main X-Lens processing pipeline.

    Qwen generates all answers.

    RAG supplies optional factual context when the user's question
    is relevant to the indexed knowledge base.
    """

    def __init__(
        self,
        quality_analyzer: Any,
        vlm: Any = None,
        rag_service: Any = None,
    ) -> None:
        self.quality = quality_analyzer
        self.vlm = vlm
        self.rag = rag_service
        self.router = RequestRouter()

        # Wearable output limits.
        self.max_speech_words = 38
        self.max_display_chars = 42

        # Qwen generation limits.
        self.text_max_new_tokens = 64
        self.vision_max_new_tokens = 36

        # RAG limits.
        self.max_rag_context_chars = 1200
        self.max_returned_chunks = 3

    def process(
        self,
        image: Any = None,
        question: str | None = None,
    ) -> dict[str, Any]:
        """
        Process a text-only or image-based request.
        """

        timing: dict[str, float] = {}

        route = self.router.classify(
            question=question,
            image_provided=image is not None,
        )

        if route.mode == RequestMode.TEXT:
            return self._process_text(
                question=route.question,
                timing=timing,
                route_reason=route.reason,
            )

        return self._process_vision(
            image=image,
            question=route.question,
            timing=timing,
            route_reason=route.reason,
        )

    def _process_text(
        self,
        question: str | None,
        timing: dict[str, float],
        route_reason: str,
    ) -> dict[str, Any]:
        """
        Answer a text-only question using Qwen.
        """

        if not question:
            return self._message_response(
                mode=RequestMode.TEXT.value,
                route_reason=route_reason,
                message="Please ask a question.",
                display_text="Please ask a question",
                timing=timing,
            )

        if self.vlm is None:
            return self._message_response(
                mode=RequestMode.TEXT.value,
                route_reason=route_reason,
                message="The Qwen model is unavailable.",
                display_text="Qwen unavailable",
                timing=timing,
            )

        context, hits = self._retrieve_context(
            question=question,
            timing=timing,
        )

        if context:
            prompt = self._build_grounded_text_prompt(
                question=question,
                context=context,
            )
            knowledge_source = "rag_and_qwen"
        else:
            prompt = self._build_general_text_prompt(
                question=question,
            )
            knowledge_source = "qwen"

        try:
            with timer(timing, "qwen_inference_ms"):
                result = self.vlm.generate_text(
                    prompt=prompt,
                    max_new_tokens=self.text_max_new_tokens,
                )
        except Exception as exc:
            return self._model_error_response(
                mode=RequestMode.TEXT.value,
                route_reason=route_reason,
                timing=timing,
                error=exc,
            )

        full_answer = self._finalize_answer(result.text)
        speech_text = self._speech_text(full_answer)
        display_text = self._display_text(full_answer)

        return {
            "status": "completed",
            "mode": RequestMode.TEXT.value,
            "knowledge_source": knowledge_source,
            "route_reason": route_reason,
            "quality": None,
            "answer": full_answer,
            "speech_text": speech_text,
            "display_text": display_text,
            "rag_chunks": self._compact_hits(hits),
            "timing": timing,
            "tokens": {
                "input": result.input_tokens,
                "output": result.output_tokens,
            },
        }

    def _process_vision(
        self,
        image: Any,
        question: str | None,
        timing: dict[str, float],
        route_reason: str,
    ) -> dict[str, Any]:
        """
        Answer an image-based question using Qwen vision.
        """

        if image is None:
            return self._message_response(
                mode=RequestMode.VISION.value,
                route_reason=route_reason,
                message="A camera image is required.",
                display_text="Camera image required",
                timing=timing,
            )

        with timer(timing, "quality_analysis_ms"):
            report = self.quality.analyze(image)

        if not report.is_acceptable:
            message = "Please capture the image again."

            return {
                "status": "recapture",
                "mode": RequestMode.VISION.value,
                "knowledge_source": "none",
                "route_reason": route_reason,
                "quality": report.to_dict(),
                "answer": message,
                "speech_text": message,
                "display_text": "Recapture image",
                "rag_chunks": [],
                "timing": timing,
                "tokens": None,
            }

        if self.vlm is None:
            return {
                "status": "error",
                "mode": RequestMode.VISION.value,
                "knowledge_source": "none",
                "route_reason": route_reason,
                "quality": report.to_dict(),
                "answer": "The Qwen model is unavailable.",
                "speech_text": "The Qwen model is unavailable.",
                "display_text": "Qwen unavailable",
                "rag_chunks": [],
                "timing": timing,
                "tokens": None,
            }

        context = ""
        hits: list[dict[str, Any]] = []

        if question:
            context, hits = self._retrieve_context(
                question=question,
                timing=timing,
            )

        prompt = self._build_vision_prompt(
            question=question,
            context=context,
        )

        knowledge_source = (
            "image_rag_and_qwen"
            if context
            else "image_and_qwen"
        )

        try:
            with timer(timing, "qwen_inference_ms"):
                result = self.vlm.generate_vision(
                    image=image,
                    prompt=prompt,
                    max_new_tokens=self.vision_max_new_tokens,
                )
        except Exception as exc:
            return {
                "status": "error",
                "mode": RequestMode.VISION.value,
                "knowledge_source": knowledge_source,
                "route_reason": route_reason,
                "quality": report.to_dict(),
                "answer": "I could not process the image.",
                "speech_text": "I could not process the image.",
                "display_text": "Image processing failed",
                "rag_chunks": self._compact_hits(hits),
                "timing": timing,
                "tokens": None,
                "error": str(exc),
            }

        full_answer = self._finalize_vision_answer(
            text=result.text,
            output_tokens=result.output_tokens,
            token_limit=self.vision_max_new_tokens,
        )

        speech_text = self._speech_text(full_answer)
        display_text = self._display_text(full_answer)

        return {
            "status": "completed",
            "mode": RequestMode.VISION.value,
            "knowledge_source": knowledge_source,
            "route_reason": route_reason,
            "quality": report.to_dict(),
            "answer": full_answer,
            "speech_text": speech_text,
            "display_text": display_text,
            "rag_chunks": self._compact_hits(hits),
            "timing": timing,
            "tokens": {
                "input": result.input_tokens,
                "output": result.output_tokens,
            },
        }

    def _retrieve_context(
        self,
        question: str,
        timing: dict[str, float],
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Retrieve optional knowledge-base context.
        """

        if self.rag is None or not question:
            return "", []

        try:
            with timer(timing, "rag_retrieval_ms"):
                context, hits = self.rag.context_for(question)
        except Exception:
            return "", []

        if not hits:
            return "", []

        selected_hits = hits[: self.max_returned_chunks]

        context_parts: list[str] = []

        for hit in selected_hits:
            text = self._clean_context(
                hit.get("text", "")
            )

            if not text:
                continue

            source = hit.get(
                "source_url",
                "unknown",
            )

            context_parts.append(
                f"Source: {source}\n{text}"
            )

        compact_context = "\n\n".join(context_parts)

        return (
            compact_context[: self.max_rag_context_chars],
            selected_hits,
        )

    def _build_general_text_prompt(
        self,
        question: str,
    ) -> str:
        """
        Build a concise prompt for general text questions.
        """

        return (
            "You are an assistant for AI smart glasses.\n"
            "Answer directly and accurately in one or two complete "
            "sentences.\n"
            "Keep the response concise and state uncertainty when needed.\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    def _build_grounded_text_prompt(
        self,
        question: str,
        context: str,
    ) -> str:
        """
        Build a prompt using retrieved knowledge-base information.
        """

        return (
            "You are an assistant for AI smart glasses.\n"
            "Answer using only facts supported by the supplied information.\n"
            "If the information is insufficient, say what cannot be "
            "confirmed.\n"
            "Answer specific questions precisely and broad questions with "
            "a concise overview.\n"
            "Use no more than two complete sentences.\n\n"
            f"Information:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    def _build_vision_prompt(
        self,
        question: str | None,
        context: str,
    ) -> str:
        """
        Build a concise vision prompt focused on accuracy,
        completeness and cautious text recognition.
        """

        if question:
            task = question.strip()
        else:
            prompt_path = Path(
                "prompts/scene_description.txt"
            )

            if prompt_path.exists():
                task = prompt_path.read_text(
                    encoding="utf-8"
                ).strip()
            else:
                task = "Describe the main visible object or scene."

        instructions = (
            "Inspect the whole image carefully.\n"
            "Answer using visible evidence only.\n"
            "Give one concise and grammatically complete sentence.\n"
            "Identify the main object, material, surface, text, or scene.\n"
            "Keep separate objects separate and do not invent connections.\n"
            "Read visible text only when reasonably clear.\n"
            "When text is uncertain, say it appears to read something "
            "instead of presenting it as exact.\n"
            "Do not use quotation marks around uncertain text.\n"
            "If identification is uncertain, state that briefly.\n"
        )

        if context:
            instructions += (
                "Use the information below only when directly relevant.\n"
                "Do not treat retrieved information as visual proof.\n"
                f"Information:\n{context}\n"
            )

        return (
            f"{instructions}\n"
            f"Question: {task}\n"
            "Answer:"
        )

    def _speech_text(
        self,
        text: str,
    ) -> str:
        """
        Produce concise speech without cutting a sentence midway.
        """

        clean = self._finalize_answer(text)

        if not clean:
            return "I could not produce an answer."

        words = clean.split()

        if len(words) <= self.max_speech_words:
            return clean

        sentences = self._split_sentences(clean)

        selected_sentences: list[str] = []
        selected_word_count = 0

        for sentence in sentences:
            sentence_words = sentence.split()

            if (
                selected_sentences
                and selected_word_count + len(sentence_words)
                > self.max_speech_words
            ):
                break

            if (
                not selected_sentences
                and len(sentence_words) > self.max_speech_words
            ):
                return self._shorten_single_sentence(
                    sentence
                )

            selected_sentences.append(sentence)
            selected_word_count += len(sentence_words)

        if selected_sentences:
            return self._ensure_sentence_ending(
                " ".join(selected_sentences)
            )

        return self._shorten_single_sentence(clean)

    def _shorten_single_sentence(
        self,
        text: str,
    ) -> str:
        """
        Shorten one long sentence at a natural punctuation boundary.
        """

        words = text.split()

        if len(words) <= self.max_speech_words:
            return self._ensure_sentence_ending(text)

        shortened = " ".join(
            words[: self.max_speech_words]
        )

        possible_endings = [
            shortened.rfind(","),
            shortened.rfind(";"),
            shortened.rfind(":"),
        ]

        natural_end = max(possible_endings)

        if natural_end >= len(shortened) // 2:
            shortened = shortened[:natural_end]

        shortened = shortened.rstrip(
            " ,;:-([{"
        )

        return self._ensure_sentence_ending(shortened)

    def _display_text(
        self,
        text: str,
    ) -> str:
        """
        Create short display text for the glasses screen.
        """

        clean = self._finalize_answer(text)

        if not clean:
            return "No answer"

        if len(clean) <= self.max_display_chars:
            return clean

        shortened = clean[
            : self.max_display_chars - 1
        ].rstrip()

        if " " in shortened:
            shortened = shortened.rsplit(
                " ",
                1,
            )[0]

        shortened = shortened.rstrip(
            " ,;:-([{"
        )

        return shortened + "…"

    def _compact_hits(
        self,
        hits: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Return compact source metadata.
        """

        return [
            {
                "source_url": hit.get("source_url"),
                "similarity": hit.get("similarity"),
            }
            for hit in hits[: self.max_returned_chunks]
        ]

    def _message_response(
        self,
        mode: str,
        route_reason: str,
        message: str,
        display_text: str,
        timing: dict[str, float],
    ) -> dict[str, Any]:
        """
        Build a standard non-model response.
        """

        return {
            "status": "completed",
            "mode": mode,
            "knowledge_source": "none",
            "route_reason": route_reason,
            "quality": None,
            "answer": message,
            "speech_text": message,
            "display_text": display_text,
            "rag_chunks": [],
            "timing": timing,
            "tokens": None,
        }

    def _model_error_response(
        self,
        mode: str,
        route_reason: str,
        timing: dict[str, float],
        error: Exception,
    ) -> dict[str, Any]:
        """
        Return a consistent Qwen error response.
        """

        message = "I could not generate an answer."

        return {
            "status": "error",
            "mode": mode,
            "knowledge_source": "none",
            "route_reason": route_reason,
            "quality": None,
            "answer": message,
            "speech_text": message,
            "display_text": "Answer generation failed",
            "rag_chunks": [],
            "timing": timing,
            "tokens": None,
            "error": str(error),
        }

    @staticmethod
    def _clean_context(
        context: str,
    ) -> str:
        """
        Normalize one retrieved context chunk.
        """

        return re.sub(
            r"\s+",
            " ",
            context or "",
        ).strip()

    @classmethod
    def _finalize_answer(
        cls,
        text: str,
    ) -> str:
        """
        Clean generated output and ensure normal punctuation.
        """

        clean = cls._clean_generated_answer(text)

        if not clean:
            return "I could not produce an answer."

        return cls._ensure_sentence_ending(clean)

    @classmethod
    def _finalize_vision_answer(
        cls,
        text: str,
        output_tokens: int,
        token_limit: int,
    ) -> str:
        """
        Clean a vision answer.

        If generation reaches its token limit and produces an unfinished
        final sentence, the incomplete final sentence is removed.
        """

        clean = cls._clean_generated_answer(text)

        if not clean:
            return "I could not identify the image clearly."

        sentences = cls._split_sentences(clean)
        reached_limit = output_tokens >= token_limit - 1

        if reached_limit and len(sentences) > 1:
            last_sentence = sentences[-1].strip()
            last_words = last_sentence.rstrip(
                ".!?"
            ).split()

            weak_endings = {
                "a",
                "an",
                "the",
                "and",
                "or",
                "with",
                "without",
                "small",
                "large",
                "some",
                "another",
                "possibly",
            }

            weak_starts = {
                "there",
                "this",
                "that",
                "it",
                "the",
                "a",
                "an",
                "which",
                "with",
            }

            looks_incomplete = False

            if not last_words:
                looks_incomplete = True
            else:
                first_word = last_words[0].lower()
                last_word = last_words[-1].lower()

                looks_incomplete = (
                    last_word in weak_endings
                    or (
                        first_word in weak_starts
                        and len(last_words) <= 5
                    )
                )

            if looks_incomplete:
                clean = " ".join(
                    sentences[:-1]
                ).strip()

        if not clean:
            return "I could not identify the image clearly."

        return cls._ensure_sentence_ending(clean)

    @staticmethod
    def _clean_generated_answer(
        text: str,
    ) -> str:
        """
        Normalize generated text and repair unmatched quotation marks.
        """

        clean = re.sub(
            r"\s+",
            " ",
            text or "",
        ).strip()

        clean = re.sub(
            r"^[\-\*\d.)\s]+",
            "",
            clean,
        )

        # Remove unmatched quotation marks.
        if clean.count('"') % 2 != 0:
            clean = clean.replace('"', "")

        # Remove an unmatched single quotation mark only when there is
        # exactly one. This avoids malformed output while preserving
        # normal contractions in most answers.
        if clean.count("'") == 1:
            clean = clean.replace("'", "")

        return clean

    @staticmethod
    def _split_sentences(
        text: str,
    ) -> list[str]:
        """
        Split text into sentences.
        """

        sentences = re.findall(
            r"[^.!?]+[.!?]+|[^.!?]+$",
            text,
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    @staticmethod
    def _ensure_sentence_ending(
        text: str,
    ) -> str:
        """
        Ensure text ends with punctuation.
        """

        clean = text.strip()

        if not clean:
            return clean

        if clean[-1] not in ".!?":
            clean += "."

        return clean