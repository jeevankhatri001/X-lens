from pydantic import BaseModel, Field
class TimingInfo(BaseModel):
    quality_analysis_ms: float=0; vlm_inference_ms: float|None=None; rag_retrieval_ms: float|None=None; total_ms: float|None=None
class ErrorResponse(BaseModel): detail:str; request_id:str|None=None
