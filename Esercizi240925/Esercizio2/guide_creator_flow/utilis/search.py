from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    PayloadSchemaType,
    FieldCondition,
    MatchValue,
    MatchText,
    Filter,
    SearchParams,
    PointStruct,
)

from typing import List, Tuple, Any
import numpy as np

class Search():

    def __init__(self, client, settings, query, embeddings):
        self.client=client
        self.settings = settings
        self.query = query
        self.embeddings=embeddings


    def qdrant_semantic_search(self,
        limit: int,
        with_vectors: bool = False
    ):
        qv = self.embeddings.embed_query(self.query)
        res = self.client.query_points(
            collection_name=self.settings.collection,
            query=qv,
            limit=limit,
            with_payload=True,
            with_vectors=with_vectors,
            search_params=SearchParams(
                hnsw_ef=256,  # ampiezza lista in fase di ricerca (recall/latency)
                exact=False   # True = ricerca esatta (lenta); False = ANN HNSW
            ),
        )
        return res.points

    def qdrant_text_prefilter_ids(self, max_hits: int) -> List[int]:
        matched_ids: List[int] = []
        next_page = None
        while True:
            points, next_page = self.client.scroll(
                collection_name=self.settings.collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="text", match=MatchText(text=self.query))]
                ),
                limit=min(256, max_hits - len(matched_ids)),
                offset=next_page,
                with_payload=False,
                with_vectors=False,
            )
            matched_ids.extend([p.id for p in points])
            if not next_page or len(matched_ids) >= max_hits:
                break
        return matched_ids

    def mmr_select(self,
        query_vec: List[float],
        candidates_vecs: List[List[float]],
        k: int,
        lambda_mult: float
    ) -> List[int]:
        
        V = np.array(candidates_vecs, dtype=float)
        q = np.array(query_vec, dtype=float)

        def cos(a, b):
            na = (a @ a) ** 0.5 + 1e-12
            nb = (b @ b) ** 0.5 + 1e-12
            return float((a @ b) / (na * nb))

        sims = [cos(v, q) for v in V]
        selected: List[int] = []
        remaining = set(range(len(V)))

        while len(selected) < min(k, len(V)):
            if not selected:
                # pick the highest similarity first
                best = max(remaining, key=lambda i: sims[i])
                selected.append(best)
                remaining.remove(best)
                continue
            best_idx = None
            best_score = -1e9
            for i in remaining:
                max_div = max([cos(V[i], V[j]) for j in selected]) if selected else 0.0
                score = lambda_mult * sims[i] - (1 - lambda_mult) * max_div
                if score > best_score:
                    best_score = score
                    best_idx = i
            selected.append(best_idx)
            remaining.remove(best_idx)
        return selected

    def hybrid_search(self):
        
        sem = self.qdrant_semantic_search(limit=self.settings.top_n_semantic, with_vectors=True)
        if not sem:
            return []

        # (2) full-text prefilter (id)
        text_ids = set(self.qdrant_text_prefilter_ids(self.settings.top_n_text))

        # Normalizzazione score semantici per fusione
        scores = [p.score for p in sem]
        smin, smax = min(scores), max(scores)
        def norm(x):  # robusto al caso smin==smax
            return 1.0 if smax == smin else (x - smin) / (smax - smin)

        # (3) fusione con boost testuale
        fused: List[Tuple[int, float, Any]] = []  # (idx, fused_score, point)
        for idx, p in enumerate(sem):
            base = norm(p.score)                    # [0..1]
            fuse = self.settings.alpha * base
            if p.id in text_ids:
                fuse += self.settings.text_boost         # boost additivo
            fused.append((idx, fuse, p))

        # ordina per fused_score desc
        fused.sort(key=lambda t: t[1], reverse=True)

        # MMR opzionale per diversificare i top-K
        if self.settings.use_mmr:
            qv = self.embeddings.embed_query(self.query)
            # prendiamo i primi N dopo fusione (es. 30) e poi MMR per final_k
            N = min(len(fused), max(self.settings.final_k * 5, self.settings.final_k))
            cut = fused[:N]
            vecs = [sem[i].vector for i, _, _ in cut]
            mmr_idx = self.mmr_select(qv, vecs, self.settings.final_k, self.settings.mmr_lambda)
            picked = [cut[i][2] for i in mmr_idx]
            return picked

        # altrimenti, prendi i primi final_k dopo fusione
        return [p for _, _, p in fused[:self.settings.final_k]]