import json
from pathlib import Path
import numpy as np
class VectorStore:
    def __init__(self,path:Path): self.path=path; self.index=None; self.metadata=[]
    def build(self,vectors:np.ndarray,metadata:list[dict]):
        try: import faiss
        except ImportError as exc: raise RuntimeError("Install RAG dependencies: pip install -e '.[rag]'") from exc
        vectors=np.asarray(vectors,dtype='float32'); self.index=faiss.IndexFlatIP(vectors.shape[1]); self.index.add(vectors); self.metadata=metadata
        self.path.mkdir(parents=True,exist_ok=True); faiss.write_index(self.index,str(self.path/'index.faiss')); (self.path/'metadata.json').write_text(json.dumps(metadata,indent=2))
    def load(self):
        import faiss
        self.index=faiss.read_index(str(self.path/'index.faiss')); self.metadata=json.loads((self.path/'metadata.json').read_text())
    def search(self,query_vector,k=5):
        if self.index is None: self.load()
        scores,ids=self.index.search(np.asarray(query_vector,dtype='float32').reshape(1,-1),k)
        return [{**self.metadata[i],"similarity":float(s)} for s,i in zip(scores[0],ids[0]) if i>=0]
