from typing import List, Tuple, Optional
from synaptoroute.models import Route

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", threshold: float = 0.5):
        if CrossEncoder is None:
            raise ImportError("sentence-transformers is not installed. Please install it using `pip install synaptoroute[rerank]`")
        self.model = CrossEncoder(model_name)
        self.threshold = threshold

    def rerank(self, query: str, candidates: List[Tuple[float, Route]]) -> Optional[Route]:
        """
        Takes the top retrieved routes (and their previous scores) and runs a cross-encoder to rerank them.
        Returns the best route if it passes the reranker's threshold, else None.
        """
        if not candidates:
            return None

        # Build pairs of (query, utterance) for all utterances in the candidate routes
        pairs = []
        route_mapping = [] # maps pair index back to a route
        
        for _, route in candidates:
            for utt in route.utterances:
                pairs.append((query, utt))
                route_mapping.append(route)
                
        if not pairs:
            return None
            
        # Score the pairs
        scores = self.model.predict(pairs)
        
        # Find the route with the maximum score
        best_score = float('-inf')
        best_route = None
        
        # Determine entailment index if 2D
        entailment_idx = 1
        if len(scores.shape) > 1 and scores.shape[1] > 1:
            if hasattr(self.model.model, 'config') and hasattr(self.model.model.config, 'label2id'):
                for label, idx in self.model.model.config.label2id.items():
                    if label.lower() == 'entailment':
                        entailment_idx = idx
                        break
        
        for i, s in enumerate(scores):
            # Extract 1D score
            score = s[entailment_idx] if len(scores.shape) > 1 else s
            
            if score > best_score:
                best_score = score
                best_route = route_mapping[i]
                
        # Compare against the cross-encoder's threshold
        if best_score >= self.threshold:
            return best_route
            
        return None
