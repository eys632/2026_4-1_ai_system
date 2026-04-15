from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_ENERGY_FILTER_KEYWORDS = (
    "에너지",
    "바우처",
    "전기",
    "도시가스",
    "가스",
    "요금",
    "난방",
    "냉방",
    "보일러",
    "에어컨",
    "조명",
    "LED",
    "신재생",
    "태양광",
    "태양열",
    "지열",
    "연료전지",
    "풍력",
    "캐시백",
)

from .config import Settings
from .embeddings import E5Embedder
from .llm import HFGenerator
from .prompts import build_rag_prompt
from .types import RAGAnswer, RetrievalResult
from .vectorstore import FaissChunkIndex


@dataclass
class RAGSystem:
    settings: Settings
    index_dir: Path
    embedder: E5Embedder
    generator: HFGenerator
    index: FaissChunkIndex

    @classmethod
    def load(cls, settings: Settings, chunker_name: str) -> "RAGSystem":
        idx_dir = settings.index_dir / chunker_name
        index = FaissChunkIndex.load(idx_dir)
        embedder = E5Embedder(model_id=settings.embedding_model_id)
        generator = HFGenerator(model_id=settings.answer_model_id, device=settings.device)
        return cls(settings=settings, index_dir=idx_dir, embedder=embedder, generator=generator, index=index)

    def answer(self, question: str) -> RAGAnswer:
        q_emb = self.embedder.embed_query(question)
        is_energy_question = "에너지" in question

        # Retrieve more candidates when we intend to post-process (filter/diversify).
        candidate_k = self.settings.top_k
        if is_energy_question:
            candidate_k = max(self.settings.top_k * 5, 30)

        results = self.index.search(q_emb, top_k=candidate_k)

        # Post-retrieval: filter + diversify for energy-related questions.
        if is_energy_question and results:
            filtered = [
                r
                for r in results
                if any(k in (r.chunk.text or "") for k in _ENERGY_FILTER_KEYWORDS)
            ]
            pool = filtered if len(filtered) >= 4 else results

            # Diversify by keyword groups so we don't end up with many near-duplicate chunks.
            keyword_groups: list[tuple[str, tuple[str, ...]]] = [
                ("바우처", ("바우처",)),
                ("캐시백", ("캐시백",)),
                ("신재생", ("신재생", "태양광", "태양열", "지열", "연료전지", "풍력")),
                ("조명", ("조명", "LED")),
                ("효율개선", ("에너지효율", "보일러", "에어컨", "난방", "냉방", "단열", "창호", "바닥")),
                ("요금", ("전기요금", "도시가스", "요금")),
            ]

            # Sort by score desc once for stable best-pick.
            pool_sorted = sorted(pool, key=lambda x: x.score, reverse=True)

            picked: list[RetrievalResult] = []
            picked_ids: set[str] = set()

            def pick_first_matching(kws: tuple[str, ...]) -> None:
                for r in pool_sorted:
                    cid = r.chunk.chunk_id
                    if cid in picked_ids:
                        continue
                    txt = r.chunk.text or ""
                    if any(k in txt for k in kws):
                        picked.append(r)
                        picked_ids.add(cid)
                        return

            for _, kws in keyword_groups:
                if len(picked) >= self.settings.top_k:
                    break
                pick_first_matching(kws)

            # Fill remaining slots by best score.
            if len(picked) < self.settings.top_k:
                for r in pool_sorted:
                    cid = r.chunk.chunk_id
                    if cid in picked_ids:
                        continue
                    picked.append(r)
                    picked_ids.add(cid)
                    if len(picked) >= self.settings.top_k:
                        break

            results = picked[: self.settings.top_k]
        else:
            results = results[: self.settings.top_k]

        contexts = [(r.chunk.source, r.chunk.title, r.chunk.text) for r in results]
        prompt = build_rag_prompt(question, contexts)
        text = self.generator.generate(prompt, max_new_tokens=self.settings.max_new_tokens, temperature=0.0)
        if "(근거:" not in text:
            used = ",".join(str(i) for i in range(1, len(contexts) + 1))
            text = (text.rstrip() + f"\n(근거: {used})").strip()
        return RAGAnswer(answer=text, contexts=results, model_id=self.settings.answer_model_id)
